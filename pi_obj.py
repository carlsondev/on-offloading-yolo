import time

import cv2
import numpy as np
import socket

from typing import List, Tuple, Optional
from utils import RectType, segment_image, recv_from_socket, send_data, select_roi, output_file_data

from onboard import setup_model, detect_frame


class Pi:
    def __init__(
        self,
        video_path: str,
        should_offload: bool,
        offload_ip_addr: str,
        offload_port: int,
    ):

        self._video_path = video_path
        self._should_offload = should_offload
        self._frame_segments_list: List[Tuple[int, List[RectType]]] = []
        self._yolo_model: Optional[cv2.dnn_Net] = None
        self._layer_names: List[str] = []

        self._weights_path = "yolov7_deps/yolov7-tiny.weights"
        self._config_path = "yolov7_deps/yolov7-tiny.cfg"

        self._offload_sock: Optional[socket.socket] = None
        self._offload_resolution = (720, 480)

        if should_offload:
            self._offload_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._offload_sock.connect((offload_ip_addr, offload_port))
            except TimeoutError as e:
                print(
                    f"Connection to offload device at {offload_ip_addr}:{offload_port} timed out!"
                )
                raise e

    def exec(self):

        video_reader = cv2.VideoCapture(self._video_path)

        got_frame, bgr_frame = video_reader.read()  # Make sure we can read video

        if not got_frame:
            print("Cannot read from video source")
            exit(1)

        self._frame_segments_list: List[Tuple[int, List[RectType]]] = [
            segment_image(bgr_frame.shape, 2),  # Split into 4ths
            # segment_image(bgr_frame.shape, 3),  # Split into 9ths
        ]

        # Pi will never use CUDA
        self._yolo_model, self._layer_names = setup_model(self._config_path, self._weights_path, False)

        curr_frame_num = 1
        while got_frame:
            self.process_frame(bgr_frame, curr_frame_num)
            got_frame, bgr_frame = video_reader.read()
            curr_frame_num += 1

        video_reader.release()

        cv2.destroyAllWindows()

    def process_frame(self, bgr_frame: np.ndarray, frame_num: int):
        """
        Process the given video frame. If self._should_offload is True, the frame will be segmented and those segments will be filtered.
        Those filtered segments will then be sent to the server which will process them and send back the results (currently only class IDs).
        If self._should_offload is False, local YOLO detection will occur.
        :param bgr_frame:
        :type bgr_frame:
        :return:
        :rtype:
        """
        if not self._should_offload:
            start = time.time()
            output_list = detect_frame(bgr_frame, self._yolo_model, self._layer_names, 0.9)
            proc_time = time.time() - start
            print(f"Frame processing took {proc_time:.2f} seconds")
            self.handle_detected_objects(output_list, proc_time, frame_num)
            return

        # Start offloaded computation
        offload_frame_list: List[np.ndarray] = []
        offload_start = time.time()
        for segment_count, img_segments in self._frame_segments_list:
            should_off, frame = self.should_offload_frame(bgr_frame, img_segments)
            if frame is not None:
                offload_frame_list.append(frame)

        for off_frame in offload_frame_list:

            # Send frame
            did_succeed = send_data(
                self._offload_sock,
                cv2.resize(off_frame, self._offload_resolution),
                "=L",
            )
            if not did_succeed:
                print("Failed to offload frame!")
                continue

            # Receive detected class IDs for the frame sent
            detected_class_ids: Optional[List[int]] = recv_from_socket(self._offload_sock, "=B", 64)

            if detected_class_ids is None:
                print("Failed to receive class IDs")
                continue

            # TODO: Temporary since we are only sending the class_ids
            obj_data = [(class_id, 1.0, (0, 0, 0, 0)) for class_id in detected_class_ids]
            offload_proc_time = time.time() - offload_start
            self.handle_detected_objects(obj_data, offload_proc_time, frame_num)

    def handle_detected_objects(
        self, detected_objects: List[Tuple[int, float, RectType]], proc_time: float, frame_num: int
    ):
        """
        Process the detected objects (whether they are received or local)
        :param detected_objects: Objects list
        """

        output_file_data(frame_num, [class_id for class_id, _, _ in detected_objects], proc_time)
        if len(detected_objects) == 0:
            print("Did not detect any objects in frame")

            return

        for (class_id, conf, rect) in detected_objects:
            print(f"Detected class {class_id} in frame")

    def should_offload_frame(
        self, bgr_frame: np.ndarray, img_segments: List[RectType]
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process the BGR frame of a video and decide whether to offload the specific frame, segment by segment

        :param bgr_frame: The frame to process
        :param img_segments: The list of image segments (x,y,w,h) to process
        :return: Whether to offload the segment or not (bool) and the segment to offboard
        """

        selected_img = select_roi(bgr_frame, img_segments)
        return True, selected_img
