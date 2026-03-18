# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torchvision
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
    parser = argparse.ArgumentParser(description="Process directories for EgoBlur detection on specific video files")
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="Face model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="License plate model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.3,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling",
    )

    parser.add_argument(
        "--input_directory",
        required=True,
        type=str,
        help="Root directory to search for video files recursively",
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
        help="Show what would be processed without actually running detections",
    )

    parser.add_argument(
        "--force_reprocess",
        required=False,
        action="store_true",
        help="Force reprocessing of videos even if output JSONL files already exist",
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> argparse.Namespace:
    """
    Run some basic checks on the input arguments
    """
    # input args value checks
    if not 0.0 <= args.face_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid face_model_score_threshold {args.face_model_score_threshold}"
        )
    if not 0.0 <= args.lp_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid lp_model_score_threshold {args.lp_model_score_threshold}"
        )
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError(f"Invalid nms_iou_threshold {args.nms_iou_threshold}")
    if not 0 <= args.scale_factor_detections:
        raise ValueError(
            f"Invalid scale_factor_detections {args.scale_factor_detections}"
        )

    # input/output paths checks
    if args.face_model_path is None and args.lp_model_path is None:
        raise ValueError(
            "Please provide either face_model_path or lp_model_path or both"
        )
    if not os.path.exists(args.input_directory):
        raise ValueError(f"Input directory {args.input_directory} does not exist.")
    if args.face_model_path is not None and not os.path.exists(args.face_model_path):
        raise ValueError(f"{args.face_model_path} does not exist.")
    if args.lp_model_path is not None and not os.path.exists(args.lp_model_path):
        raise ValueError(f"{args.lp_model_path} does not exist.")

    return args


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        if not torch.cuda.is_available()
        else f"cuda:{torch.cuda.current_device()}"
    )


def find_target_videos(root_directory: str, logger: logging.Logger) -> List[Tuple[str, str]]:
    """
    Find all aria_fused.mp4 and thinklet_fused.mp4 files recursively
    Returns list of tuples: (video_path, output_jsonl_path)
    """
    target_files = ["aria_fused.mp4", "thinklet_fused.mp4"]
    found_videos = []
    
    logger.info(f"Searching for target videos in: {root_directory}")
    logger.info(f"Target files: {target_files}")
    
    root_path = Path(root_directory)
    
    for target_file in target_files:
        # Search recursively for the target file
        for video_path in root_path.rglob(target_file):
            # Create output path next to the video file
            video_dir = video_path.parent
            output_name = target_file.replace("_fused.mp4", "_egoblur_detection.jsonl")
            output_path = video_dir / output_name
            
            found_videos.append((str(video_path), str(output_path)))
            logger.info(f"Found: {video_path} -> {output_path}")
    
    logger.info(f"Total videos found: {len(found_videos)}")
    return found_videos


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format
    """
    bgr_image = cv2.imread(image_path)
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections
    Return the image tensor
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())
    return image_tensor


def get_detections(
    detector: torch.jit._script.RecursiveScriptModule,
    image_tensor: torch.Tensor,
    model_score_threshold: float,
    nms_iou_threshold: float,
) -> Tuple[List[List[float]], List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections and scores
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]
    scores = scores[score_keep_idx]
    
    return boxes.tolist(), scores.tolist()


def scale_box(
    box: List[float], max_width: int, max_height: int, scale: float
) -> List[float]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def collect_detection_data(
    image: np.ndarray,
    face_detections: List[List[float]],
    face_scores: List[float],
    lp_detections: List[List[float]],
    lp_scores: List[float],
    scale_factor_detections: float,
    frame_index: int = None,
) -> Dict[str, Any]:
    """
    Collect detection data without applying blurring
    """
    image_height, image_width = image.shape[:2]
    
    detection_data = {
        "image_width": image_width,
        "image_height": image_height,
        "scale_factor_detections": scale_factor_detections,
        "face_detections": [],
        "license_plate_detections": []
    }
    
    if frame_index is not None:
        detection_data["frame_index"] = frame_index
    
    # Process face detections
    for i, (box, score) in enumerate(zip(face_detections, face_scores)):
        if scale_factor_detections != 1.0:
            scaled_box = scale_box(box, image_width, image_height, scale_factor_detections)
        else:
            scaled_box = box
            
        detection_data["face_detections"].append({
            "detection_id": i,
            "bbox": {
                "x1": float(scaled_box[0]),
                "y1": float(scaled_box[1]),
                "x2": float(scaled_box[2]),
                "y2": float(scaled_box[3])
            },
            "original_bbox": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            },
            "confidence_score": float(score),
            "detection_type": "face"
        })
    
    # Process license plate detections
    for i, (box, score) in enumerate(zip(lp_detections, lp_scores)):
        if scale_factor_detections != 1.0:
            scaled_box = scale_box(box, image_width, image_height, scale_factor_detections)
        else:
            scaled_box = box
            
        detection_data["license_plate_detections"].append({
            "detection_id": i,
            "bbox": {
                "x1": float(scaled_box[0]),
                "y1": float(scaled_box[1]),
                "x2": float(scaled_box[2]),
                "y2": float(scaled_box[3])
            },
            "original_bbox": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            },
            "confidence_score": float(score),
            "detection_type": "license_plate"
        })
    
    return detection_data


def process_video(
    input_video_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Perform detections on the input video and return detection data for all frames.
    """
    logger.info(f"Processing video: {input_video_path}")
    start_time = time.time()
    
    video_detection_data = []
    video_reader_clip = VideoFileClip(input_video_path)
    
    total_frames = int(video_reader_clip.fps * video_reader_clip.duration)
    logger.info(f"Video info: {total_frames} frames, {video_reader_clip.fps:.2f} FPS, {video_reader_clip.duration:.2f}s duration")
    
    # Create progress bar for frame processing
    frame_progress = tqdm(
        enumerate(video_reader_clip.iter_frames()),
        total=total_frames,
        desc=f"Processing {Path(input_video_path).name}",
        unit="frames"
    )
    
    for frame_index, frame in frame_progress:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_tensor = get_image_tensor(bgr_image)
        image_tensor_copy = image_tensor.clone()
        
        face_detections, face_scores = [], []
        lp_detections, lp_scores = [], []
        
        # get face detections
        if face_detector is not None:
            face_detections, face_scores = get_detections(
                face_detector,
                image_tensor,
                face_model_score_threshold,
                nms_iou_threshold,
            )
        
        # get license plate detections
        if lp_detector is not None:
            lp_detections, lp_scores = get_detections(
                lp_detector,
                image_tensor_copy,
                lp_model_score_threshold,
                nms_iou_threshold,
            )
        
        detection_data = collect_detection_data(
            bgr_image,
            face_detections,
            face_scores,
            lp_detections,
            lp_scores,
            scale_factor_detections,
            frame_index,
        )
        
        detection_data["input_video_path"] = input_video_path
        video_detection_data.append(detection_data)

    video_reader_clip.close()
    frame_progress.close()
    
    total_time = time.time() - start_time
    avg_fps = len(video_detection_data) / total_time
    logger.info(f"Video processing completed!")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average processing speed: {avg_fps:.2f} FPS")
    logger.info(f"Processed {len(video_detection_data)} frames")
    
    return video_detection_data


def save_jsonl(data: List[Dict[str, Any]], output_path: str, logger: logging.Logger) -> None:
    """
    Save detection data to JSONL file
    """
    logger.info(f"Saving detection data to: {output_path}")
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Successfully saved {len(data)} detection records")


def process_directory(
    args: argparse.Namespace,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    logger: logging.Logger,
) -> None:
    """
    Main function to process all videos in the directory
    """
    # Find all target videos
    video_files = find_target_videos(args.input_directory, logger)
    
    if not video_files:
        logger.warning("No target videos found!")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual processing will be performed")
        for video_path, output_path in video_files:
            logger.info(f"Would process: {video_path} -> {output_path}")
        return
    
    # Process each video
    overall_progress = tqdm(
        video_files,
        desc="Processing videos",
        unit="video"
    )
    
    successful_processing = 0
    failed_processing = 0
    skipped_processing = 0
    
    for video_path, output_path in overall_progress:
        try:
            # Check if output already exists
            if os.path.exists(output_path) and not args.force_reprocess:
                logger.warning(f"Output file already exists: {output_path}. Skipping... (use --force_reprocess to overwrite)")
                skipped_processing += 1
                continue
            elif os.path.exists(output_path) and args.force_reprocess:
                logger.info(f"Output file exists but force reprocessing enabled: {output_path}. Reprocessing...")
            
            # Process the video
            video_detection_data = process_video(
                video_path,
                face_detector,
                lp_detector,
                args.face_model_score_threshold,
                args.lp_model_score_threshold,
                args.nms_iou_threshold,
                args.scale_factor_detections,
                logger,
            )
            
            # Save results
            save_jsonl(video_detection_data, output_path, logger)
            successful_processing += 1
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {str(e)}")
            failed_processing += 1
    
    overall_progress.close()
    
    # Final summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
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
        
        logger.info("Starting EgoBlur directory processing")
        logger.info(f"Input directory: {args.input_directory}")
        logger.info(f"Face model: {args.face_model_path}")
        logger.info(f"License plate model: {args.lp_model_path}")
        logger.info(f"Device: {get_device()}")
        
        # Load models
        face_detector = None
        lp_detector = None
        
        if args.face_model_path is not None:
            logger.info("Loading face detector...")
            face_detector = torch.jit.load(args.face_model_path, map_location="cpu").to(get_device())
            face_detector.eval()
            logger.info("Face detector loaded successfully")
        
        if args.lp_model_path is not None:
            logger.info("Loading license plate detector...")
            lp_detector = torch.jit.load(args.lp_model_path, map_location="cpu").to(get_device())
            lp_detector.eval()
            logger.info("License plate detector loaded successfully")
        
        # Process directory
        process_directory(args, face_detector, lp_detector, logger)
        
        logger.info("Directory processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
