#!/usr/bin/env python3


import sys

import cv2 as cv
import numpy as np


def gstreamer_pipeline(
    sensor_id=1,
    capture_width=3840,
    capture_height=2160,
    display_width=3840,
    display_height=2160,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


video_capture = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)
ret_val, frame = video_capture.read()
sys.stdout.buffer.write(frame.tobytes())
