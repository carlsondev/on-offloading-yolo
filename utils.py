from typing import Tuple, List, Optional, Any
import socket
import time
import struct
import pickle
import cv2
import numpy as np
import os

# (x, y, w, h)
RectType = Tuple[float, float, float, float]
out_file_path = "results.txt"


def output_file_data(frame_num: int, class_list: List[int], proc_time: float):

    # If file already exists, delete
    if frame_num == 1 and os.path.exists(out_file_path):
        os.remove(out_file_path)

    with open(out_file_path, "a") as out_file:
        if frame_num == 1:
            # Add header
            out_file.write("frame_num,people_detected_count,processing_time")

        # Add row
        detected_people = class_list.count(0)
        row = [frame_num, detected_people, proc_time]
        out_file.write(",".join(map(str, row)))


def select_roi(img, img_rects: List[RectType]) -> np.ndarray:
    value_list = []
    image_list = []
    for (x, y, w, h) in img_rects:
        cropped_image = img[y : y + h, x : x + w]
        temp = cropped_image.copy()
        image_list.append(cropped_image)
        temp[:, :, 1] = 0
        gray_img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray_img)
        value_list.append(mean)
    image_index = np.argwhere(value_list == np.max(value_list))
    return image_list[image_index[0][0]]


def segment_image(img_shape: Tuple[float, float], segment_count: int) -> Tuple[int, List[RectType]]:
    """
    Generate rects for sections separating the image into nxn sections

    :param img_shape: (w, h) shape of the image
    :param segment_count: N segment count.
    :return: NxN list of rects for sections of the image with the segment count that
     is passed in (used for transforming back to original space)
    """
    img_h, img_w, _ = img_shape
    # Generate image rects
    img_rects: List[RectType] = []

    x_seg_w = img_w // segment_count
    y_seg_h = img_h // segment_count

    for x_seg_idx in range(segment_count):
        curr_seg_x = x_seg_idx * x_seg_w
        current_seg_w = (img_w - curr_seg_x) - ((segment_count - x_seg_idx - 1) * x_seg_w)

        for y_seg_idx in range(segment_count):
            curr_seg_y = y_seg_idx * y_seg_h
            current_seg_h = (img_h - curr_seg_y) - ((segment_count - y_seg_idx - 1) * y_seg_h)
            img_rects.append((curr_seg_x, curr_seg_y, current_seg_w, current_seg_h))

    return segment_count, img_rects


def send_data(sock: socket.socket, data: Any, header_format: str) -> bool:
    """
    Convert data to bytes and send via socket with the data size (in bytes) pre-appended (as a header)

    :param sock: The socker to send the data on
    :param data: The data to send
    :param header_format: The format of the header (for example: "=B" or "=L")
    :return: True if the sending was successful, False otherwise
    """
    # Converts input to byte data
    byte_data = pickle.dumps(data)

    # Gets data length as header format (long, byte, etc)
    print(f"Byte Data len for type ({header_format}): {len(byte_data)}")
    message_size = struct.pack(header_format, len(byte_data))

    try:
        # Sends message size as a header followed by data
        # If None, the request succeeded
        return sock.sendall(message_size + byte_data) is None
    except TimeoutError:
        print(f"Failed to send frame to offloading device with timeout!")
        return False


def recv_from_socket(
    sock: socket.socket, header_format: str, byte_recv_count: int
) -> Optional[Any]:
    """
    Receive data from the specified socket

    :param sock: The socket to recieve data from
    :param header_format: The format of the header data (for example: "=B" or "=L")
    :param byte_recv_count: The max amount of bytes to read during each recv call.
    :return: The received value. None if no value received.
    """
    socket_data = b""
    payload_head_size = struct.calcsize(header_format)

    start = time.time()

    # Receive message size
    while len(socket_data) < payload_head_size:
        socket_data += sock.recv(byte_recv_count)

    # Convert message size data to header format (byte, long, etc)
    packed_msg_size = socket_data[:payload_head_size]
    msg_size = struct.unpack(header_format, packed_msg_size)[0]

    # Shift socket data past header length
    socket_data = socket_data[payload_head_size:]

    # Receive the rest of the message
    while len(socket_data) < msg_size:
        socket_data += sock.recv(byte_recv_count)

    print(f"Recv took: {(time.time() - start)}")

    # Convert message byte data back to original data
    frame_data = socket_data[:msg_size]
    frame: Optional[Any] = pickle.loads(frame_data)

    return frame
