import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    return cap