# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time
from functools import lru_cache
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
import torchvision
from moviepy.video.io.VideoFileClip import VideoFileClip

REPORT_INTERVAL = 1000


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--input_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given image on which we want to make detections",
    )

    parser.add_argument(
        "--output_jsonl_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the detection results as JSONL",
    )

    parser.add_argument(
        "--input_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given video on which we want to make detections",
    )

    parser.add_argument(
        "--output_video_jsonl_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the video detection results as JSONL",
    )

    return parser.parse_args()


def create_output_directory(file_path: str) -> None:
    """
    parameter file_path: absolute path to the directory where we want to create the output files
    Simple logic to create output directories if they don't exist.
    """
    print(
        f"Directory {os.path.dirname(file_path)} does not exist. Attempting to create it..."
    )
    os.makedirs(os.path.dirname(file_path))
    if not os.path.exists(os.path.dirname(file_path)):
        raise ValueError(
            f"Directory {os.path.dirname(file_path)} didn't exist. Attempt to create also failed. Please provide another path."
        )


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """
    parameter args: parsed arguments
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
    if args.input_image_path is None and args.input_video_path is None:
        raise ValueError("Please provide either input_image_path or input_video_path")
    if args.input_image_path is not None and args.output_jsonl_path is None:
        raise ValueError(
            "Please provide output_jsonl_path for the detection results to save."
        )
    if args.input_video_path is not None and args.output_video_jsonl_path is None:
        raise ValueError(
            "Please provide output_video_jsonl_path for the video detection results to save."
        )
    if args.input_image_path is not None and not os.path.exists(args.input_image_path):
        raise ValueError(f"{args.input_image_path} does not exist.")
    if args.input_video_path is not None and not os.path.exists(args.input_video_path):
        raise ValueError(f"{args.input_video_path} does not exist.")
    if args.face_model_path is not None and not os.path.exists(args.face_model_path):
        raise ValueError(f"{args.face_model_path} does not exist.")
    if args.lp_model_path is not None and not os.path.exists(args.lp_model_path):
        raise ValueError(f"{args.lp_model_path} does not exist.")
    if args.output_jsonl_path is not None and not os.path.exists(
        os.path.dirname(args.output_jsonl_path)
    ):
        create_output_directory(args.output_jsonl_path)
    if args.output_video_jsonl_path is not None and not os.path.exists(
        os.path.dirname(args.output_video_jsonl_path)
    ):
        create_output_directory(args.output_video_jsonl_path)

    # check we have write permissions on output paths
    if args.output_jsonl_path is not None and not os.access(
        os.path.dirname(args.output_jsonl_path), os.W_OK
    ):
        raise ValueError(
            f"You don't have permissions to write to {args.output_jsonl_path}. Please grant adequate permissions, or provide a different output path."
        )
    if args.output_video_jsonl_path is not None and not os.access(
        os.path.dirname(args.output_video_jsonl_path), os.W_OK
    ):
        raise ValueError(
            f"You don't have permissions to write to {args.output_video_jsonl_path}. Please grant adequate permissions, or provide a different output path."
        )

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
) -> List[List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections
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
    
    # Return both boxes and scores for data collection
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
    parameter image: image on which we want to make detections
    parameter face_detections: list of face bounding boxes in format [x1, y1, x2, y2]
    parameter face_scores: list of face detection scores
    parameter lp_detections: list of license plate bounding boxes in format [x1, y1, x2, y2]
    parameter lp_scores: list of license plate detection scores
    parameter scale_factor_detections: scale detections by the given factor
    parameter frame_index: frame index for video processing (None for images)

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


def process_image(
    input_image_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
) -> Dict[str, Any]:
    """
    parameter input_image_path: absolute path to the input image
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: license plate detector model to perform license plate detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter scale_factor_detections: scale detections by the given factor

    Perform detections on the input image and return detection data.
    """
    print(f"Starting image processing: {input_image_path}")
    start_time = time.time()
    
    bgr_image = read_image(input_image_path)
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
    )
    
    detection_data["input_image_path"] = input_image_path
    
    processing_time = time.time() - start_time
    print(f"Image processing completed in {processing_time:.3f} seconds")
    print(f"Found {len(face_detections)} face detections and {len(lp_detections)} license plate detections")
    
    return detection_data


def process_video(
    input_video_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
) -> List[Dict[str, Any]]:
    """
    parameter input_video_path: absolute path to the input video
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: license plate detector model to perform license plate detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter scale_factor_detections: scale detections by the given factor

    Perform detections on the input video and return detection data for all frames.
    """
    print(f"Starting video processing: {input_video_path}")
    start_time = time.time()
    
    video_detection_data = []
    video_reader_clip = VideoFileClip(input_video_path)
    
    total_frames = int(video_reader_clip.fps * video_reader_clip.duration)
    print(f"Video info: {total_frames} frames, {video_reader_clip.fps:.2f} FPS, {video_reader_clip.duration:.2f}s duration")
    
    frame_start_time = time.time()
    
    for frame_index, frame in enumerate(video_reader_clip.iter_frames()):
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
        
        # Progress reporting every REPORT_INTERVAL frames
        if (frame_index + 1) % REPORT_INTERVAL == 0:
            elapsed_time = time.time() - frame_start_time
            fps_processed = (frame_index + 1) / elapsed_time
            remaining_frames = total_frames - (frame_index + 1)
            eta_seconds = remaining_frames / fps_processed if fps_processed > 0 else 0
            print(f"Processed {frame_index + 1}/{total_frames} frames ({fps_processed:.2f} FPS) - ETA: {eta_seconds:.1f}s")

    video_reader_clip.close()
    
    total_time = time.time() - start_time
    avg_fps = len(video_detection_data) / total_time
    print(f"Video processing completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average processing speed: {avg_fps:.2f} FPS")
    print(f"Processed {len(video_detection_data)} frames")
    
    return video_detection_data


def save_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    parameter data: list of detection data dictionaries
    parameter output_path: absolute path where we want to save the JSONL file
    """
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    args = validate_inputs(parse_args())
    
    if args.face_model_path is not None:
        face_detector = torch.jit.load(args.face_model_path, map_location="cpu").to(
            get_device()
        )
        face_detector.eval()
    else:
        face_detector = None

    if args.lp_model_path is not None:
        lp_detector = torch.jit.load(args.lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

    if args.input_image_path is not None:
        detection_data = process_image(
            args.input_image_path,
            face_detector,
            lp_detector,
            args.face_model_score_threshold,
            args.lp_model_score_threshold,
            args.nms_iou_threshold,
            args.scale_factor_detections,
        )
        save_jsonl([detection_data], args.output_jsonl_path)
        print(f"Detection data saved to {args.output_jsonl_path}")

    if args.input_video_path is not None:
        video_detection_data = process_video(
            args.input_video_path,
            face_detector,
            lp_detector,
            args.face_model_score_threshold,
            args.lp_model_score_threshold,
            args.nms_iou_threshold,
            args.scale_factor_detections,
        )
        save_jsonl(video_detection_data, args.output_video_jsonl_path)
        print(f"Video detection data saved to {args.output_video_jsonl_path}")
