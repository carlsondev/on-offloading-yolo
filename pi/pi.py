import cv2
import numpy as np
import sys
from typing import Tuple, List

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


def process_frame(bgr_frame: np.ndarray, img_segments: List[RectType]):

    for (x, y, w, h) in img_segments:
        segmented_img = bgr_frame[y : y + h, x : x + w]

        # Apply HOG


def main(file_path: str):

    video_reader = cv2.VideoCapture(file_path)

    got_frame, bgr_frame = video_reader.read()  # Make sure we can read video

    if not got_frame:
        print("Cannot read from video source")
        exit(1)

    frame_segments_list: List[Tuple[int, List[RectType]]] = [
        segment_image(bgr_frame.shape, 2),  # Split into 4ths
        segment_image(bgr_frame.shape, 3),  # Split into 9ths
    ]

    while got_frame:

        got_frame, bgr_frame = video_reader.read()
        for segment_count, img_segments in frame_segments_list:
            process_frame(bgr_frame, img_segments)

    video_reader.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Need to supply the input video file path")
        exit(1)

    args = sys.argv[1:]

    main(args[0])
