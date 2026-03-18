# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply blurring to videos based on JSONL detection data with dilation support")
    parser.add_argument(
        "--input_directory",
        required=True,
        type=str,
        help="Root directory to search for video files recursively",
    )

    parser.add_argument(
        "--dilation_suffix",
        required=True,
        type=str,
        help="Dilation suffix for input JSONL files (e.g., 'dilate_5' for aria_egoblur_detection_dilate_5.jsonl)",
    )

    parser.add_argument(
        "--output_dilation_suffix",
        required=False,
        type=str,
        default="",
        help="Dilation suffix for output video files (e.g., 'dilate_5' for aria_blurred_dilate_5.mp4). If not provided, output will be aria_blurred.mp4",
    )

    parser.add_argument(
        "--output_video_fps",
        required=False,
        type=int,
        default=30,
        help="FPS for the output blurred videos",
    )

    parser.add_argument(
        "--log_level",
        required=False,
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--dry_run",
        required=False,
        action="store_true",
        help="Show what would be processed without actually running blurring",
    )

    parser.add_argument(
        "--force_overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing blurred videos",
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> argparse.Namespace:
    """
    Run some basic checks on the input arguments
    """
    if not os.path.exists(args.input_directory):
        raise ValueError(f"Input directory {args.input_directory} does not exist.")
    
    if not 1 <= args.output_video_fps or not (
        isinstance(args.output_video_fps, int) and args.output_video_fps % 1 == 0
    ):
        raise ValueError(
            f"Invalid output_video_fps {args.output_video_fps}, should be a positive integer"
        )

    return args


def find_target_videos_with_jsonl(
    root_directory: str, 
    dilation_suffix: str,
    output_dilation_suffix: str,
    logger: logging.Logger
) -> List[Tuple[str, str, str]]:
    """
    Find all aria_fused.mp4 and thinklet_fused.mp4 files that have corresponding JSONL files
    Returns list of tuples: (video_path, jsonl_path, output_video_path)
    """
    target_files = ["aria_fused.mp4", "thinklet_fused.mp4"]
    found_videos = []
    
    logger.info(f"Searching for target videos with JSONL data in: {root_directory}")
    logger.info(f"Target files: {target_files}")
    logger.info(f"Using dilation suffix for JSONL: {dilation_suffix}")
    logger.info(f"Using output dilation suffix: {output_dilation_suffix if output_dilation_suffix else '(none - default naming)'}")
    
    root_path = Path(root_directory)
    
    for target_file in target_files:
        # Search recursively for the target file
        for video_path in root_path.rglob(target_file):
            # Check for corresponding JSONL file with dilation suffix
            video_dir = video_path.parent
            jsonl_name = target_file.replace("_fused.mp4", f"_egoblur_detection_{dilation_suffix}.jsonl")
            jsonl_path = video_dir / jsonl_name
            
            # Create output video path
            if output_dilation_suffix:
                output_name = target_file.replace("_fused.mp4", f"_blurred_{output_dilation_suffix}.mp4")
            else:
                output_name = target_file.replace("_fused.mp4", "_blurred.mp4")
            output_path = video_dir / output_name
            
            if jsonl_path.exists():
                found_videos.append((str(video_path), str(jsonl_path), str(output_path)))
                logger.info(f"Found: {video_path} -> {jsonl_path} -> {output_path}")
            else:
                logger.warning(f"Skipping {video_path}: No JSONL file found at {jsonl_path}")
    
    logger.info(f"Total videos with JSONL data found: {len(found_videos)}")
    return found_videos


def load_jsonl_detections(jsonl_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load detection data from JSONL file
    """
    logger.debug(f"Loading detection data from: {jsonl_path}")
    detections = []
    
    try:
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        detection_data = json.loads(line)
                        detections.append(detection_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON on line {line_num}: {e}")
                        continue
        
        logger.info(f"Loaded {len(detections)} detection records from {jsonl_path}")
        return detections
        
    except FileNotFoundError:
        logger.error(f"JSONL file not found: {jsonl_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading JSONL file {jsonl_path}: {e}")
        return []


def extract_detections_from_frame_data(frame_data: Dict[str, Any]) -> List[List[float]]:
    """
    Extract bounding boxes from frame detection data
    """
    detections = []
    
    # Add face detections
    for face_detection in frame_data.get("face_detections", []):
        bbox = face_detection["bbox"]
        detections.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
    
    # Add license plate detections
    for lp_detection in frame_data.get("license_plate_detections", []):
        bbox = lp_detection["bbox"]
        detections.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
    
    return detections


def apply_blurring(
    image: np.ndarray,
    detections: List[List[float]],
    logger: logging.Logger,
) -> np.ndarray:
    """
    Apply blurring to image based on detection bounding boxes
    Uses the same technique as the original demo_ego_blur.py
    """
    if not detections:
        return image
    
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    for box in detections:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1

        # Use the same blur kernel size as original
        ksize = (image.shape[0] // 2, image.shape[1] // 2)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image


def process_video_with_detections(
    input_video_path: str,
    jsonl_path: str,
    output_video_path: str,
    output_fps: int,
    logger: logging.Logger,
) -> bool:
    """
    Process video and apply blurring based on JSONL detection data
    Memory-efficient version that processes frames one at a time
    """
    logger.info(f"Processing video: {input_video_path}")
    logger.info(f"Using detection data: {jsonl_path}")
    logger.info(f"Output video: {output_video_path}")
    
    start_time = time.time()
    
    # Load detection data
    detection_data = load_jsonl_detections(jsonl_path, logger)
    if not detection_data:
        logger.error(f"No detection data loaded from {jsonl_path}")
        return False
    
    # Create a mapping from frame index to detections
    frame_detections = {}
    for frame_data in detection_data:
        frame_index = frame_data.get("frame_index", 0)
        detections = extract_detections_from_frame_data(frame_data)
        frame_detections[frame_index] = detections
    
    logger.info(f"Loaded detections for {len(frame_detections)} frames")
    
    # Process video
    video_reader_clip = VideoFileClip(input_video_path)
    total_frames = int(video_reader_clip.fps * video_reader_clip.duration)
    logger.info(f"Video info: {total_frames} frames, {video_reader_clip.fps:.2f} FPS, {video_reader_clip.duration:.2f}s duration")
    
    # Set up video writer using OpenCV for memory efficiency
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (video_reader_clip.w, video_reader_clip.h)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, frame_size)
    
    if not video_writer.isOpened():
        logger.error(f"Failed to initialize video writer for {output_video_path}")
        video_reader_clip.close()
        return False
    
    # Create progress bar for frame processing
    frame_progress = tqdm(
        enumerate(video_reader_clip.iter_frames()),
        total=total_frames,
        desc=f"Blurring {Path(input_video_path).name}",
        unit="frames"
    )
    
    processed_frames = 0
    
    for frame_index, frame in frame_progress:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Get detections for this frame
        detections = frame_detections.get(frame_index, [])
        
        # Apply blurring
        blurred_frame = apply_blurring(frame, detections, logger)
        
        # Convert RGB to BGR for OpenCV
        blurred_frame_bgr = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame directly to video file
        video_writer.write(blurred_frame_bgr)
        processed_frames += 1
    
    # Clean up
    video_reader_clip.close()
    video_writer.release()
    frame_progress.close()
    
    total_time = time.time() - start_time
    logger.info(f"Video blurring completed in {total_time:.2f} seconds")
    logger.info(f"Processed {processed_frames} frames")
    logger.info(f"Saved blurred video to: {output_video_path}")
    
    return True


def process_directory(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """
    Main function to process all videos in the directory
    """
    # Find all target videos with JSONL data
    video_files = find_target_videos_with_jsonl(
        args.input_directory, 
        args.dilation_suffix,
        args.output_dilation_suffix,
        logger
    )
    
    if not video_files:
        logger.warning("No target videos with JSONL data found!")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual processing will be performed")
        for video_path, jsonl_path, output_path in video_files:
            logger.info(f"Would process: {video_path} -> {output_path}")
        return
    
    # Process each video
    overall_progress = tqdm(
        video_files,
        desc="Blurring videos",
        unit="video"
    )
    
    successful_processing = 0
    failed_processing = 0
    skipped_processing = 0
    
    for video_path, jsonl_path, output_path in overall_progress:
        try:
            # Check if output already exists
            if os.path.exists(output_path) and not args.force_overwrite:
                logger.warning(f"Output video already exists: {output_path}. Skipping... (use --force_overwrite to overwrite)")
                skipped_processing += 1
                continue
            
            # Process the video
            success = process_video_with_detections(
                video_path,
                jsonl_path,
                output_path,
                args.output_video_fps,
                logger,
            )
            
            if success:
                successful_processing += 1
            else:
                failed_processing += 1
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {str(e)}")
            failed_processing += 1
    
    overall_progress.close()
    
    # Final summary
    logger.info("=" * 60)
    logger.info("BLURRING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos found: {len(video_files)}")
    logger.info(f"Successfully processed: {successful_processing}")
    logger.info(f"Failed processing: {failed_processing}")
    logger.info(f"Skipped (already exists): {skipped_processing}")


if __name__ == "__main__":
    # Parse arguments and setup logging
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    try:
        # Validate inputs
        args = validate_inputs(args, logger)
        
        logger.info("Starting EgoBlur video blurring with dilation support")
        logger.info(f"Input directory: {args.input_directory}")
        logger.info(f"Dilation suffix: {args.dilation_suffix}")
        logger.info(f"Output dilation suffix: {args.output_dilation_suffix if args.output_dilation_suffix else '(none - default naming)'}")
        logger.info(f"Output video FPS: {args.output_video_fps}")
        logger.info(f"Force overwrite: {args.force_overwrite}")
        
        # Process directory
        process_directory(args, logger)
        
        logger.info("Video blurring completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise





