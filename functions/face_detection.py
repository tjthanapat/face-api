import numpy as np
import cv2
from typing import List, Tuple, Union
from deepface import DeepFace
from .custom_classes import FaceObj
import os


def detect_mtcnn(
    img: Union[np.ndarray,str],
    target_size: Tuple[int, int],
) -> List[FaceObj]:
    """Detect and align faces in a given image with MTCNN.

    Parameters
    ----------
    img : array with shape (H, W, C) or str
        Image array or image path
    target_size : Tuple[int, int]
        Target size of cropped face region(s).

    Returns
    -------
    face_objs : list of face objects
        Each face object contains face (cropped and aligned face image),
        facial_area (x, y, w, h) and confidence.
    """
    try:
        face_objs = DeepFace.extract_faces(
            img,
            target_size=target_size,
            detector_backend="mtcnn",
        )
        face_objs_ = [
            dict(
                face=face_obj["face"],
                facial_area=face_obj["facial_area"],
                # DeepFace returns confidence in numpy.float64
                confidence=face_obj["confidence"].item(),
            )
            for face_obj in face_objs
        ]
        return face_objs_
    except ValueError:
        # DeepFace.extract_faces will raise ValueError exception when no face is detected.
        return []

_face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")


def detect_opencv(
    img: np.ndarray,
    target_size: Tuple[int, int],
) -> List[FaceObj]:
    """Detect faces in a given image with OpenCV's Haar Cascade

    Parameters
    ----------
    img : array with shape (H, W, C)
    target_size : Tuple[int, int]
        Target size of cropped face region(s).

    Returns
    -------
    face_objs : list of face objects
        Each face object contains face (cropped face image),
        facial_area (x, y, w, h) and confidence.
    """
    faces = _face_cascade.detectMultiScale3(img, 1.1, 10, outputRejectLevels=True)
    # faces = (List[bbox], List[rejectLevels], List[rejectLevels])
    face_objs = list()

    for i in range(len(faces[0])):
        (x, y, w, h) = faces[0][i].tolist()
        face_img = img[y : y + h, x : x + w, :].copy()
        face_img = cv2.resize(face_img, target_size)
        face_obj = dict(
            face=face_img,
            facial_area=dict(
                x=x,
                y=y,
                w=w,
                h=h,
            ),
            confidence=faces[2].ravel()[i],
        )
        face_objs.append(face_obj)

    face_objs = sorted(
        face_objs,
        key=lambda face_obj: face_obj["confidence"],
        reverse=True,
    )
    return face_objs