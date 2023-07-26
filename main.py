import tensorflow as tf

# # Turn off interactive logging
# tf.keras.utils.disable_interactive_logging()
# # Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
import logging
import os

from model.facenet import load_facenet_model
import functions.face_detection as face_detection
import functions.face_embedding as face_embedding
import functions.face_recognition as face_recognition
import functions.manage_db as manage_db
import functions.utilities as utilities
from functions.custom_api_responses import API_RESPONSES, API_STATUS_CODE

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import List
from functions.custom_classes import DetectionObj, VerificationObj, RecognitionObj


APP_TITLE = "Face API"
APP_VERSION = "1.2.0"
APP_DESCRIPTION = """
Face API - Detection, Verification, Recognition.

- Supported image file types: jpg, png
- Available detection model: MTCNN, OpenCV's Haar Cascade
- Face embedding model: FaceNet
"""
VERBOSE = True  # Flag for api logging


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)


# load facenet model
weight_path = "model/facenet_weights.h5"
model = load_facenet_model(weight_path)


# create db directory
DB_PATH = "db"
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(f"{DB_PATH}/images", exist_ok=True)


@app.on_event("startup")
async def startup_event():
    ##### Run model inferrence testing on starting api service #####
    logging.info("Running inferrence testing")
    print("Running inferrence testing")
    faces = face_detection.detect_mtcnn(
        "test_image.png",
        target_size=model.input_shape[1:3],
    )
    embedding = face_embedding.embed(
        faces[0]["face"],
        embedding_model=model,
    )
    del faces
    del embedding


@app.get("/")
def read_root():
    return "Face API. Visit `/docs` to use api swagger."


@app.post(
    "/detect/mtcnn",
    tags=["detect"],
    response_model=List[DetectionObj],
    responses={
        # fmt: off
        API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"]: API_RESPONSES["NOT_SUPPORTED_IMAGE_FILE"],
        API_STATUS_CODE["FACE_NOT_DETECTED"]: API_RESPONSES["FACE_NOT_DETECTED"],
        # fmt: on
    },
)
async def detect_faces_mtcnn(
    imgFile: UploadFile,
):
    """Detecting face(s) in a given image with MTCNN."""
    if VERBOSE:
        logging.info("Recieved /detect/mtcnn request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    img = await utilities.read_img_file(imgFile)

    if VERBOSE:
        logging.info("Detecting faces in an image")
    faces = face_detection.detect_mtcnn(
        img,
        target_size=model.input_shape[1:3],
    )
    if len(faces) == 0:
        error_message = "Face could not be detected in an image."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["FACE_NOT_DETECTED"],
            detail=error_message,
        )

    faces_ = [
        dict(
            detectionConfidence=face_obj["confidence"],
            area=face_obj["facial_area"],
        )
        for face_obj in faces
    ]
    return faces_


@app.post(
    "/detect/opencv",
    tags=["detect"],
    response_model=List[DetectionObj],
    responses={
        # fmt: off
        API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"]: API_RESPONSES["NOT_SUPPORTED_IMAGE_FILE"],
        API_STATUS_CODE["FACE_NOT_DETECTED"]: API_RESPONSES["FACE_NOT_DETECTED"],
        # fmt: on
    },
)
async def detect_faces_opencv(
    imgFile: UploadFile,
):
    """Detecting face(s) in a given image with OpenCV's Haar Cascade.
    \nNote: Detection confidence is not in range 0-1.
    """
    if VERBOSE:
        logging.info("Recieved /detect/opencv request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    img = await utilities.read_img_file(imgFile)

    if VERBOSE:
        logging.info("Detecting faces in an image")
    faces = face_detection.detect_opencv(
        img,
        target_size=model.input_shape[1:3],
    )
    if len(faces) == 0:
        error_message = "Face could not be detected in an image."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["FACE_NOT_DETECTED"],
            detail=error_message,
        )

    faces_ = [
        dict(
            detectionConfidence=face_obj["confidence"],
            area=face_obj["facial_area"],
        )
        for face_obj in faces
    ]
    return faces_


@app.post(
    "/verify",
    tags=["verify"],
    response_model=VerificationObj,
    responses={
        # fmt: off
        API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"]: API_RESPONSES["NOT_SUPPORTED_IMAGE_FILE"],
        API_STATUS_CODE["FACE_NOT_DETECTED"]: API_RESPONSES["FACE_NOT_DETECTED"],
        # fmt: on
    },
)
async def verify_face(
    imgFileToVerify: UploadFile,
    imgFileAuthentic: UploadFile,
    detection: Literal["mtcnn", "opencv"] = "mtcnn",
):
    """Note: In case there is more than one face detected in image,
    only one with the highest confidence is used for verification."""
    if VERBOSE:
        logging.info("Recieved /verify request")
        logging.info(
            "Reading image files "
            + f'(To-Verify Img: "{imgFileToVerify.filename}" Authentic Img: "{imgFileAuthentic.filename}")'
        )
    imgFileToVerify = await utilities.read_img_file(imgFileToVerify)
    imgFileAuthentic = await utilities.read_img_file(imgFileAuthentic)

    if detection == "opencv":
        detect_faces = face_detection.detect_opencv
    else:
        detect_faces = face_detection.detect_mtcnn

    if VERBOSE:
        logging.info(f"Detecting face in a to-verify image with {detection}")
    faces_to_verify = detect_faces(
        imgFileToVerify,
        target_size=model.input_shape[1:3],
    )
    if len(faces_to_verify) == 0:
        error_message = "Face could not be detected in a to-verify image."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["FACE_NOT_DETECTED"],
            detail=error_message,
        )
    elif len(faces_to_verify) > 1:
        logging.warning(
            f"{len(faces_to_verify)} faces have been detected in a to-verify image. "
            + "Only one with the highest confidence is used for verification."
        )

    if VERBOSE:
        logging.info(f"Detecting face in an authentic image with {detection}")
    faces_authentic = detect_faces(
        imgFileAuthentic,
        target_size=model.input_shape[1:3],
    )
    if len(faces_authentic) == 0:
        error_message = "Face could not be detected in an authentic image."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["FACE_NOT_DETECTED"],
            detail=error_message,
        )
    elif len(faces_authentic) > 1:
        logging.warning(
            f"{len(faces_authentic)} faces have been detected in an authentic image. "
            + "Only one with the highest confidence is used for verification."
        )

    if VERBOSE:
        logging.info("Embedding face in a to-verify image")
    embedding_to_verify = face_embedding.embed(
        faces_to_verify[0]["face"],
        embedding_model=model,
    )
    if VERBOSE:
        logging.info("Embedding face in an authentic image")
    embedding_authentic = face_embedding.embed(
        faces_authentic[0]["face"],
        embedding_model=model,
    )

    if VERBOSE:
        logging.info("Calculating similarity")
    confidence = utilities.calculate_similarity(
        embedding_to_verify,
        embedding_authentic,
    )

    response = dict(
        confidence=confidence,
        faceToVerify=dict(
            detectionConfidence=faces_to_verify[0]["confidence"],
            area=faces_to_verify[0]["facial_area"],
        ),
        faceAuthentic=dict(
            detectionConfidence=faces_authentic[0]["confidence"],
            area=faces_authentic[0]["facial_area"],
        ),
    )
    return response


@app.post(
    "/recognize",
    tags=["recognize"],
    response_model=RecognitionObj,
    responses={
        # fmt: off
        API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"]: API_RESPONSES["NOT_SUPPORTED_IMAGE_FILE"],
        API_STATUS_CODE["FACE_NOT_DETECTED"]: API_RESPONSES["FACE_NOT_DETECTED"],
        API_STATUS_CODE["NO_DB_FILE_FOUND"]: API_RESPONSES["NO_DB_FILE_FOUND"],
        API_STATUS_CODE["NO_EMBEDDING_IN_DB"]: API_RESPONSES["NO_EMBEDDING_IN_DB"],
        # fmt: on
    },
)
async def recognize_face(
    imgFile: UploadFile,
    detection: Literal["mtcnn", "opencv"] = "mtcnn",
):
    """Note: In case there is more than one face detected in image,
    only one with the highest confidence is used for recognition."""
    if VERBOSE:
        logging.info("Recieved /recognize request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    img = await utilities.read_img_file(imgFile)

    if detection == "opencv":
        detect_faces = face_detection.detect_opencv
    else:
        detect_faces = face_detection.detect_mtcnn
    if VERBOSE:
        logging.info(f"Detecting faces in an image with {detection}")
    faces = detect_faces(
        img,
        target_size=model.input_shape[1:3],
    )
    if len(faces) == 0:
        error_message = "Face could not be detected in an image."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["FACE_NOT_DETECTED"],
            detail=error_message,
        )
    elif len(faces) > 1:
        logging.warning(
            f"{len(faces)} faces have been detected in an image. "
            + "Only one with the highest confidence would be embedded."
        )

    if VERBOSE:
        logging.info("Embedding face in an image")
    embedding = face_embedding.embed(
        faces[0]["face"],
        embedding_model=model,
    )

    if VERBOSE:
        logging.info("Getting embedding db")
    db_filepath = f"{DB_PATH}/embeddings.csv"
    db = manage_db.get_db(db_filepath)
    if len(db) == 0:
        error_message = "There is no embedding in db."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NO_EMBEDDING_IN_DB"],
            detail=error_message,
        )


    if VERBOSE:
        logging.info("Recognizing")
    recognition_results = face_recognition.recognize(
        embedding=embedding,
        db_embeddings=db,
    )
    del db
    return recognition_results[0]


@app.post(
    "/recognize/db/add",
    tags=["recognize"],
    response_model=str,
    responses={
        # fmt: off
        API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"]: API_RESPONSES["NOT_SUPPORTED_IMAGE_FILE"],
        API_STATUS_CODE["FACE_NOT_DETECTED"]: API_RESPONSES["FACE_NOT_DETECTED"],
        # fmt: on
    },
)
async def add_face_into_db(
    imgFile: UploadFile,
    subjectId: str,
    detection: Literal["mtcnn", "opencv"] = "mtcnn",
):
    """Add a new face into db. There is currently no check for duplicate id.\n
    Note: In case there is more than one face detected in image,
    only one with the highest confidence is added into db.
    """
    if VERBOSE:
        logging.info("Recieved /recognize/db/add request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    img = await utilities.read_img_file(imgFile)

    if detection == "opencv":
        detect_faces = face_detection.detect_opencv
    else:
        detect_faces = face_detection.detect_mtcnn
    if VERBOSE:
        logging.info(f"Detecting faces in an image with {detection}")
    faces = detect_faces(
        img,
        target_size=model.input_shape[1:3],
    )
    if len(faces) == 0:
        error_message = "Face could not be detected in an image."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["FACE_NOT_DETECTED"],
            detail=error_message,
        )
    elif len(faces) > 1:
        logging.warning(
            f"{len(faces)} faces have been detected in an image. "
            + "Only one with the highest confidence would be embedded."
        )

    if VERBOSE:
        logging.info("Embedding face in an image")
    embedding = face_embedding.embed(
        faces[0]["face"],
        embedding_model=model,
    )

    if VERBOSE:
        logging.info("Add embedding into db")
    db_filepath = f"{DB_PATH}/embeddings.csv"
    manage_db.add_embedding(db_filepath, subjectId, embedding)
    return f"Successfully added subject with id {subjectId}"


@app.get(
    "/recognize/db/query",
    tags=["recognize"],
    response_model=List[str],
    responses={
        # fmt: off
        API_STATUS_CODE["NO_DB_FILE_FOUND"]: API_RESPONSES["NO_DB_FILE_FOUND"],
        # fmt: on
    },
)
async def query_subjects_in_db():
    """Query all subject ids in db."""
    db_filepath = f"{DB_PATH}/embeddings.csv"
    return manage_db.query_subjects(db_filepath)


@app.get(
    "/recognize/db/remove",
    tags=["recognize"],
    response_model=str,
    responses={
        # fmt: off
        API_STATUS_CODE["NO_DB_FILE_FOUND"]: API_RESPONSES["NO_DB_FILE_FOUND"],
        API_STATUS_CODE["NO_SUBJECT_ID_IN_DB"]: API_RESPONSES["NO_SUBJECT_ID_IN_DB"],
        # fmt: on
    },
)
async def remove_subject_in_db(
    subjectId: str,
):
    """Remove all subjects with a given id in db."""
    db_filepath = f"{DB_PATH}/embeddings.csv"
    manage_db.remove_subject(db_filepath, subjectId)
    return f"Successfully removed subject(s) with id {subjectId}"


if __name__ == "__main__":
    uvicorn.run(app, port=8000, log_config="log.ini")
