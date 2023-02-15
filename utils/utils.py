from typing import Tuple, List, Optional, Any
import socket
import time
import struct
import pickle
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import torch
import torch.nn.functional as F
import math

# (x, y, w, h)
RectType = Tuple[float, float, float, float]
out_file_path = "results.txt"


def tensorify_image(image):
  img = cv2.resize(image, (352,240))
  image = torch.cuda.FloatTensor(img.transpose((2, 0, 1))).unsqueeze(0).div(255.0)
  return image


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss = torch.cuda.FloatTensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim_cuda(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
    pad = window_size // 2
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()
    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)
    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2
    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)
    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2
    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)
    if full:
        return ret, contrast_metric
    return ret


def output_file_data(frame_num: int, class_list: List[int], proc_time: float):

    # If file already exists, delete
    if frame_num == 1 and os.path.exists(out_file_path):
        os.remove(out_file_path)

    with open(out_file_path, "a") as out_file:
        if frame_num == 1:
            # Add header
            out_file.write("frame_num,people_detected_count,processing_time\n")

        # Add row
        detected_people = class_list.count(0)
        row = [frame_num, detected_people, proc_time]
        out_file.write(",".join(map(str, row))+"\n")


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


def ssim_select_cpu(image_list):
    """
    This is the cpu version of SSIM
    """
    most_dissimilar = image_list[0]
    highest_dissimilar = -1
    for i in range(len(image_list)):
        original = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
        original = cv2.resize(original, (352, 240))
        for j in range(i+1, len(image_list)):
            compare = cv2.cvtColor(image_list[j], cv2.COLOR_BGR2GRAY)
            compare = cv2.resize(compare, (352, 240))
            dissimilar = 1 - ssim(original, compare)
            #print(dissimilar)
            if dissimilar > highest_dissimilar:
                highest_dissimilar = dissimilar
                most_dissimilar = image_list[j]
    return most_dissimilar


def ssim_select_cuda(image_list):
    """
    This is the gpu version of SSIM
    """
    most_dissimilar = image_list[0]
    highest_dissimilar = -1
    for i in range(len(image_list)):
        original = tensorify_image(image_list[i])
        for j in range(i + 1, len(image_list)):
            compare = tensorify_image(image_list[j])
            dissimilar = 1 - (ssim_cuda(original, compare, val_range=255).to('cpu').tolist())
            #print(dissimilar)
            if dissimilar > highest_dissimilar:
                highest_dissimilar = dissimilar
                most_dissimilar = image_list[j]
    return most_dissimilar


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

def create_image_list(img, img_rects):
    image_list = []
    for (x,y,w,h) in img_rects[1]:
        cropped_image = img[y:y+h, x:x+w]
        image_list.append(cropped_image)
    return image_list

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
