import tensorflow as tf

# Turn off interactive logging
# tf.keras.utils.disable_interactive_logging()

# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')


import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, status

import face_verification
import face_recognition
from facenet import load_model
from utils import read_img_file, Verification, FaceObj
from api_custom_responses import API_STATUS_CODE, API_RESPONSES
import json
from typing import List
import logging

APP_TITLE = "Face API"
APP_VERSION = "1.1.0"
APP_DESCRIPTION = """
Face API - Extraction, Verification, Recognition.
"""

VERBOSE = True  # Flag for api logging


# load FaceNet model
weight_path = "facenet_weights.h5"
model = load_model(weight_path)


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

DB_PATH = "db"


@app.on_event("startup")
async def startup_event():
    ##### Run model inferrence testing on starting api service #####
    logging.info("Running inferrence testing")
    print("Running inferrence testing")
    faces = face_verification.detect(
        "test_image.png",
        target_size=model.input_shape[1:3],
    )
    embedding = face_verification.embed(
        faces[0]["face"],
        embedding_model=model,
    )
    del faces
    del embedding


@app.get("/")
def read_root():
    return "Face API. Visit `/docs` to use api swagger."


@app.post("/verify/withimage", response_model=Verification, responses=API_RESPONSES)
async def verify_face_with_image(
    imgFileToVerify: UploadFile,
    imgFileAuthentic: UploadFile,
):
    # Read image files
    if VERBOSE:
        logging.info("Recieved /verify/withimage request")
        logging.info(
            "Reading image files "
            + f'(To-Verify Img: "{imgFileToVerify.filename}" Authentic Img: "{imgFileAuthentic.filename}")'
        )
    if not imgFileToVerify.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "To-Verify image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    if not imgFileAuthentic.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Authentic image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    img_to_verify = read_img_file(await imgFileToVerify.read())
    img_authentic = read_img_file(await imgFileAuthentic.read())

    # Detect face in a to-verify image
    if VERBOSE:
        logging.info("Detecting face in a to-verify image")
    faces_to_verify = face_verification.detect(
        img_to_verify,
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

    # Detect face in an authentic image
    if VERBOSE:
        logging.info("Detecting face in an authentic image")
    faces_authentic = face_verification.detect(
        img_authentic,
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

    # Embed face images to feature vectors
    if VERBOSE:
        logging.info("Embedding face in a to-verify image")
    embedding_to_verify = face_verification.embed(
        faces_to_verify[0]["face"],
        embedding_model=model,
    )
    if VERBOSE:
        logging.info("Embedding face in an authentic image")
    embedding_authentic = face_verification.embed(
        faces_authentic[0]["face"],
        embedding_model=model,
    )

    # Calculate similarity between two embeddings
    if VERBOSE:
        logging.info("Calculating similarity")
    confidence = face_verification.calculate_similarity(
        embedding_to_verify,
        embedding_authentic,
    )

    if VERBOSE:
        logging.info("Returning response")
    response = dict(
        confidence=confidence,
        faceToVerify=dict(
            detectionConfidence=float(faces_to_verify[0]["confidence"]),
            area=faces_to_verify[0]["facial_area"],
        ),
        faceAuthentic=dict(
            detectionConfidence=float(faces_authentic[0]["confidence"]),
            area=faces_authentic[0]["facial_area"],
        ),
    )
    return response


@app.post(
    "/verify/withimage/opencv",
    response_model=Verification,
    responses=API_RESPONSES,
)
async def verify_face_with_image_opencv(
    imgFileToVerify: UploadFile,
    imgFileAuthentic: UploadFile,
):
    # Read image files
    if VERBOSE:
        logging.info("Recieved /verify/withimage request")
        logging.info(
            "Reading image files "
            + f'(To-Verify Img: "{imgFileToVerify.filename}" Authentic Img: "{imgFileAuthentic.filename}")'
        )
    if not imgFileToVerify.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "To-Verify image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    if not imgFileAuthentic.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Authentic image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    img_to_verify = read_img_file(await imgFileToVerify.read())
    img_authentic = read_img_file(await imgFileAuthentic.read())

    # Detect face in a to-verify image
    if VERBOSE:
        logging.info("Detecting face in a to-verify image")
    faces_to_verify = face_verification.detect_opencv(
        img_to_verify,
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

    # Detect face in an authentic image
    if VERBOSE:
        logging.info("Detecting face in an authentic image")
    faces_authentic = face_verification.detect(
        img_authentic,
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

    # Embed face images to feature vectors
    if VERBOSE:
        logging.info("Embedding face in a to-verify image")
    embedding_to_verify = face_verification.embed(
        faces_to_verify[0]["face"],
        embedding_model=model,
    )
    if VERBOSE:
        logging.info("Embedding face in an authentic image")
    embedding_authentic = face_verification.embed(
        faces_authentic[0]["face"],
        embedding_model=model,
    )

    # Calculate similarity between two embeddings
    if VERBOSE:
        logging.info("Calculating similarity")
    confidence = face_verification.calculate_similarity(
        embedding_to_verify,
        embedding_authentic,
    )

    if VERBOSE:
        logging.info("Returning response")
    response = dict(
        confidence=confidence,
        faceToVerify=dict(
            detectionConfidence=float(faces_to_verify[0]["confidence"]),
            area=faces_to_verify[0]["facial_area"],
        ),
        faceAuthentic=dict(
            detectionConfidence=float(faces_authentic[0]["confidence"]),
            area=faces_authentic[0]["facial_area"],
        ),
    )
    return response


@app.post("/detect", response_model=List[FaceObj], responses=API_RESPONSES)
async def detect_faces(
    imgFile: UploadFile,
):
    if VERBOSE:
        logging.info("Recieved /detect request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    if not imgFile.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    img = read_img_file(await imgFile.read())

    if VERBOSE:
        logging.info("Detecting faces in an image")
    faces = face_verification.detect(
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


@app.post("/detectandembed", response_model=str, responses=API_RESPONSES)
async def detect_and_embed_face(
    imgFile: UploadFile,
):
    if VERBOSE:
        logging.info("Recieved /detectandembed request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    if not imgFile.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    img = read_img_file(await imgFile.read())

    if VERBOSE:
        logging.info("Detecting faces in an image")
    faces = face_verification.detect(
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
    embedding = face_verification.embed(
        faces[0]["face"],
        embedding_model=model,
    )

    response = json.dumps(embedding.tolist())  # convert numpy array to list
    return response


@app.post("/recognition/add", response_model=str, responses=API_RESPONSES)
async def add_face(
    subject_id: str,
    imgFile: UploadFile,
):
    if VERBOSE:
        logging.info("Recieved /recognition/add request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    if not imgFile.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    img = read_img_file(await imgFile.read())

    if VERBOSE:
        logging.info("Detecting faces in an image")
    faces = face_verification.detect(
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
    embedding = face_verification.embed(
        faces[0]["face"],
        embedding_model=model,
    )

    if VERBOSE:
        logging.info("Adding embedding into db")
    face_recognition.add_embedding_into_db(
        db_embeddings_filepath=f"{DB_PATH}/embeddings.csv",
        subject_id=subject_id,
        embedding=embedding,
    )
    return "Successfully added."


@app.post(
    "/recognition/recognize",
    response_model=face_recognition.RecognitionResult,
    responses=API_RESPONSES,
)
async def recognize(
    imgFile: UploadFile,
):
    if VERBOSE:
        logging.info("Recieved /recognition/recognize request")
        logging.info(f'Reading image file (Img: "{imgFile.filename}")')
    if not imgFile.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        error_message = "Image file must be jpg or png format."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"],
            detail=error_message,
        )
    img = read_img_file(await imgFile.read())

    if VERBOSE:
        logging.info("Detecting faces in an image")
    faces = face_verification.detect(
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
    embedding = face_verification.embed(
        faces[0]["face"],
        embedding_model=model,
    )
    if VERBOSE:
        logging.info("Recognizing")
    recognition_results = face_recognition.recognize_face(
        embedding=embedding,
        db_embeddings_filepath=f"{DB_PATH}/embeddings.csv",
    )
    return recognition_results[0]


if __name__ == "__main__":
    uvicorn.run(app)
