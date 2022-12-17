import cv2
import numpy as np
from typing import List, Tuple
from utils import RectType


def setup_model(config_path: str, weights_path: str) -> Tuple[cv2.dnn_Net, List[str]]:
    """
    Initializes the given DARKNET model

    :param config_path: The DARKNET configuration path
    :param weights_path: The DARKNET weights path
    :return: The model network object and the output layer names
    :rtype: Tuple[cv2.dnn_Net, List[str]]
    """
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, layer_name


def detect_frame(
    bgr_frame: np.ndarray, model: cv2.dnn_Net, layers: List[str], min_conf: float
) -> List[Tuple[int, float, RectType]]:
    """
    Detects the objects in a BGR frame using YOLO

    :param bgr_frame: Frame to detect
    :param model: The YOLO model to use (opencv Net)
    :param layers: Network output layers
    :param min_conf: Confidence threshold
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

    return return_list
