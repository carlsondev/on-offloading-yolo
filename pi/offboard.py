import numpy as np
from typing import Tuple, List, Optional

# (x, y, w, h)
RectType = Tuple[float, float, float, float]


def segment_image(
    img_shape: Tuple[float, float], segment_count: int
) -> Tuple[int, List[RectType]]:
    """
    Generate rects for sections separating the image into nxn sections

    :param img_shape: (w, h) shape of the image
    :param segment_count: N segment count.
    :return: NxN list of rects for sections of the image with the segment count that
     is passed in (used for transforming back to original space)
    """
    img_h, img_w = img_shape
    # Generate image rects
    img_rects: List[RectType] = []

    x_seg_w = img_w // segment_count
    y_seg_h = img_h // segment_count

    for x_seg_idx in range(segment_count):
        curr_seg_x = x_seg_idx * x_seg_w
        current_seg_w = (img_w - curr_seg_x) - (
            (segment_count - x_seg_idx - 1) * x_seg_w
        )

        for y_seg_idx in range(segment_count):
            curr_seg_y = y_seg_idx * y_seg_h
            current_seg_h = (img_h - curr_seg_y) - (
                (segment_count - y_seg_idx - 1) * y_seg_h
            )
            img_rects.append((curr_seg_x, curr_seg_y, current_seg_w, current_seg_h))

    return segment_count, img_rects


def process_frame(
    bgr_frame: np.ndarray, img_segments: List[RectType]
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Process the BGR frame of a video, segment by segment

    :param bgr_frame: The frame to process
    :param img_segments: The list of image segments (x,y,w,h) to process
    :return: Whether to offboard the segment or not (bool) and the segment to offboard
    """
    for (x, y, w, h) in img_segments:
        segmented_img = bgr_frame[y : y + h, x : x + w]

        # Apply HOG

    return False, None
