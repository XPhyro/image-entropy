#!/usr/bin/env python3


import sys

import cv2 as cv


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
        "nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method={flip_method} ! "
        "video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx !"
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink"
    )


video_capture = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)
ret_val, frame = video_capture.read()
sys.stdout.buffer.write(frame.tobytes())
