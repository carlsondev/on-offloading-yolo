import socket
import struct
import time

import cv2
from typing import Optional

from utils.onboard import setup_model, detect_frame
from utils import recv_from_socket, send_data


class Server:
    def __init__(self, ip_addr: str, port: int):
        self._ip_addr = ip_addr
        self._port = port

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket Created")

        # Cannot catch errors easily, should be handled by OS according to docs
        self._server_sock.bind((self._ip_addr, self._port))
        print("Socket bind complete")

        self._socket_data = b""
        self._payload_head_size = struct.calcsize("=L")  # Size of long

        self._weights_path = "yolov7_deps/yolov7.weights"
        self._config_path = "yolov7_deps/yolov7.cfg"

        # Server will likely always use CUDA
        self._yolo_model, self._layer_names = setup_model(
            self._config_path, self._weights_path, True
        )

        self._active_conn: Optional[socket.socket] = None

    def execute(self):
        self._server_sock.listen(10)
        print("socket now listening...")
        self._active_conn, _ = self._server_sock.accept()

        print("Socket did accept connection")
        frame_num = 1
        while True:
            frame = recv_from_socket(self._active_conn, "=L", 4096 * 2 * 2 * 2)
            if frame is None:
                print("Streaming finished")
                return
            print(f"Current Frame Num: {frame_num}")
            self.process_frame(frame)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            frame_num += 1

    def process_frame(self, bgr_frame):
        """
        Detects the objects in the given frame and sends the class IDs back to the Pi

        :param bgr_frame: Frame to detect
        """
        start = time.time()
        output_list = detect_frame(bgr_frame, self._yolo_model, self._layer_names, 0.9)
        print(f"Frame processing took {time.time() - start:.2f} seconds")
        class_id_list = [class_id for (class_id, conf, rect) in output_list]

        # Send class IDs
        send_data(self._active_conn, class_id_list, "=H")
