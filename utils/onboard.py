import cv2
import numpy as np
from typing import List, Tuple
from utils import RectType
from cv2.dnn import DNN_BACKEND_OPENCV, DNN_BACKEND_CUDA, DNN_TARGET_CPU, DNN_TARGET_CUDA


def setup_model(
    config_path: str, weights_path: str, use_cuda: bool
) -> Tuple[cv2.dnn_Net, List[str]]:
    """
    Initializes the given DARKNET model


    :param config_path: The DARKNET configuration path
    :param weights_path: The DARKNET weights path
    :param use_cuda:  Whether to use CUDA for YOLO execution
    :return: The model network object and the output layer names
    :rtype: Tuple[cv2.dnn_Net, List[str]]
    """

    backend = DNN_BACKEND_CUDA if use_cuda else DNN_BACKEND_OPENCV
    target = DNN_TARGET_CUDA if use_cuda else DNN_TARGET_CPU

    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)

    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, layer_name


def detect_frame(
    bgr_frame: np.ndarray,
    model: cv2.dnn_Net,
    layers: List[str],
    min_conf: float,
    NMS_THRESHOLD: float = 0.3,
) -> List[Tuple[int, float, RectType]]:
    """
    Detects the objects in a BGR frame using YOLO


    :param bgr_frame: Frame to detect
    :param model: The YOLO model to use (opencv Net)
    :param layers: Network output layers
    :param min_conf: Confidence threshold
    :param NMS_THRESHOLD: Non-maxima suppression threshold
    :return List of detected (classID, confidence, RectType) tuples
    """
    h, w = bgr_frame.shape[:2]

    blob = cv2.dnn.blobFromImage(bgr_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layer_out = model.forward(layers)

    return_list: List[Tuple[int, float, RectType]] = []

    for output in layer_out:
        for detection in output:

            scores = detection[5:]
            classID: int = np.argmax(scores)
            confidence: float = scores[classID]

            if confidence <= min_conf:
                continue

            box = detection[0:4] * np.array([w, h, w, h])
            center_x, center_y, width, height = box.astype("int")

            x = center_x - (width / 2)
            y = center_y - (height / 2)

            return_list.append((classID, confidence, (x, y, width, height)))

    boxes = [box for _, _, box in return_list]
    confidences = [conf for _, conf, _ in return_list]

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, NMS_THRESHOLD)

    if len(indices) <= 0:
        return []

    return_list = [return_list[i] for i in indices.flatten()]
    return return_list
