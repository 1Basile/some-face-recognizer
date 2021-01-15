from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import logging
import argparse
import imutils
import pickle
import time
import signal
import cv2
import os
import sys

__logger = None
__showing_video = False
__video_stream_obj_link = None
__fps_obj_link = None

FRAMES_TO_ENSHURE_DETECTION = 20
__PATH_TO_FILE = os.getcwd()
__all__ = ['FRAMES_TO_ENSHURE_DETECTION', 'face_recognize']

# console usage
"""
 python3 video_recognize.py --detector face_detection_model/ \
 --embedding-model openface.nn4.small2.v1.t7 \
 --recognizer face_recognizing_model/recognizer.pickle \
 --le face_recognizing_model/le.pickle \
 -v

"""


def terminate_right_way(_signo, _stack_frame):
    """Function terminates program, and log that it happened."""
    global __logger, __showing_video, __video_stream_obj_link, __fps_obj_link
    if __logger:
        [handler.setFormatter(logging.Formatter(
            '[%(levelname)s] %(message)s'
        )) for handler in __logger.handlers]

        if __showing_video:
            cv2.destroyAllWindows()
        if __fps_obj_link:
            __fps_obj_link.stop()
        if __video_stream_obj_link:
            __video_stream_obj_link.stop()

        __logger.debug("terminated with SIGTERM code.")

    sys.exit(_signo)


def __face_recognize(detector: str, embedding_model: str, recognizer: str, le: str, confidence, max_delay,
                     showing_video, person='', logger=logging):
    """
    Inner function to do face detection and recognition.

    Parameters
    ---------
    :param detector: str
        path to OpenCV's deep learning face detector
    :param embedding_model: str
        path to OpenCV's deep learning face embedding model
    :param recognizer: str
        path to model trained to recognize faces
    :param le: str
        path to label encoder
    :param confidence: float
        minimum probability to filter weak detections in (0, 1)
    :param max_delay: default=0(float)
        if not 0, defines what delay to wait if no person infront of camera
    :param showing_video: default=False
        if video of detection should be shown
    :param person: default=''
        set whom to detect `or` empty if endless video detection
    :param logger: default=logging
        instance to catch log messages

    Return
    --------
    :return : str
        name of recognized person, or empty str if in endless detection mode(no person specified)
    """
    if not showing_video and not person:
        raise argparse.ArgumentTypeError('If you are not using video mode set person whom to detect.')

    # catching terminate signal
    signal.signal(signal.SIGTERM, terminate_right_way)

    # need to pass to terminate handler
    global __logger, __showing_video, __video_stream_obj_link, __fps_obj_link
    __logger = logger
    __showing_video = showing_video

    logger.debug("loading face detector...")
    # load serialized face detector
    protoPath = os.path.sep.join([detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector,
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    logger.debug("loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model)

    # load face recognition model with the label encoder
    recognizer = pickle.loads(open(recognizer, "rb").read())
    le = pickle.loads(open(le, "rb").read())

    if showing_video:
        logging.debug("starting video stream...")
    vs = VideoStream().start()
    __video_stream_obj_link = vs
    time.sleep(2.0)
    fps = FPS().start()
    __fps_obj_link = fps

    # for how_long is detected
    detected_time = 0
    # if no person `before` camera, slow FPS

    if showing_video:
        delta_delay = 0
    else:
        delta_delay = 0.0004

    delay = 0

    # loop over frames from the video file stream
    logging.debug("starting frames processing...")
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        if frame.any():
            (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence_ = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence_ > confidence:
                # compute box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                delay = delta_delay

                # construct a blob
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                if showing_video:
                    # draw the bounding box of the face
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                if person:
                    if name == person:
                        detected_time += 1
        else:
            if delay < max_delay:
                delay += 0.01

            # update the FPS counter
            fps.update()
            time.sleep(delay)

        if showing_video:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        if person:
            if detected_time > FRAMES_TO_ENSHURE_DETECTION:
                logger.debug("{0} recognized.".format(person))
                return person

    # stop the timer and display FPS information
    fps.stop()
    logger.debug("elasped time: {:.2f}".format(fps.elapsed()))
    logger.debug("approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    if showing_video:
        cv2.destroyAllWindows()
    vs.stop()
    return ""


def face_recognize(showing_video=False, person='', confidence=0.5, max_delay=0.5, logger=logging):
    dir_to_module = os.path.dirname(__file__)
    face_recognize.__doc__ = __face_recognize.__doc__
    face_recognize.__name__ = __face_recognize.__name__

    if person and person not in tuple(os.listdir("{}/dataset/".format(dir_to_module))):
        raise argparse.ArgumentTypeError(
            '\nWrong person name or no name specified.\nIf you are not using video mode set person whom to detect.')

    return __face_recognize(detector="{0}/face_detection_model/".format(dir_to_module),
                            embedding_model="{0}/openface.nn4.small2.v1.t7".format(dir_to_module),
                            recognizer="{0}/face_recognizing_model/recognizer.pickle".format(dir_to_module),
                            le="{0}/face_recognizing_model/le.pickle".format(dir_to_module),
                            confidence=confidence, max_delay=max_delay,
                            person=person, showing_video=showing_video,
                            logger=logger)


if __name__ == '__main__':
    logger = logging.getLogger("face_unlock_daemon")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(message)s'
    ))
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", required=True,
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", required=True,
                    help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", required=True,
                    help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", required=True,
                    help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--showing-video", action='store_true',
                    help="if video of detection should be shown")
    ap.add_argument("-p", "--person", required=False, choices=tuple(os.listdir("./dataset/")),
                    help="set whom to detect to return true if not showing video")
    ap.add_argument("-w", "--max-delay", type=float, default=0.5,
                    help="if not 0, defines what delay to wait if no person infront of camera")

    __face_recognize(**vars(ap.parse_args()), logger=logger)
