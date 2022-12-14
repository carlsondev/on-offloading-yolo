import cv2
from offboard import segment_image, process_frame, RectType
import sys
from typing import Tuple, List


def main(file_path: str, should_offload: bool):

    if not should_offload:
        print("Onboard computation currently not supported")
        exit(1)

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
        print("Usage: python pi.py path/to/video/mp4 [--onboard]")
        exit(1)

    args = sys.argv[1:]

    should_off = True
    if len(args) > 1 and args[1] == "--onboard":
        should_off = False

    main(args[0], should_off)
