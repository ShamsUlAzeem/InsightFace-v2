import cv2 as cv


def show_bboxes(img, bounding_boxes, facial_landmarks=[], wait=True):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of numpy image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
        wait: if you don't want it to wait for an input to go to the next frame
    """

    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

    cv.imshow('image', img)
    if wait:
        cv.waitKey(0)
    else:
        cv.waitKey(10)
