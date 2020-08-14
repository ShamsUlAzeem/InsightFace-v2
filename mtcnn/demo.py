import cv2 as cv
import numpy as np
from PIL import Image

from mtcnn.detector import detect_faces
from mtcnn.visualization_utils import show_bboxes

from cv2 import *


def demo_using_images():
    img = Image.open('images/0_fp_1_aligned.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/0_fp_0_aligned.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/0_fn_0_aligned.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/5_fn_0_aligned.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/6_fn_1_aligned.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)


def demo_using_webcam():
    cam = VideoCapture(0)  # set the port of the camera as before

    cam.set(cv2.CAP_PROP_FPS, 15)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret_val, img = cam.read()  # return a True boolean and and the image if all go right
        if ret_val:
            bounding_boxes, landmarks = detect_faces(Image.fromarray(img))
            show_bboxes(img, bounding_boxes, landmarks, wait=False)
        else:
            break

    cam.release()  # Closes video file or capturing device.


if __name__ == '__main__':
    demo_using_webcam()
    demo_using_images()
