import tensorflow as tf

# Turn off interactive logging
# tf.keras.utils.disable_interactive_logging()  

# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU') 


import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, status

import face_verification
from facenet import load_model
from utils import read_img_file, Verification, FaceObj
import json
from typing import List


APP_TITLE = "Face Verification API"
APP_VERSION = "1.0.2"
APP_DESCRIPTION = """
Face Verification API.
"""

VERBOSE = True  # Flag for api logging

import logging
logger = logging.getLogger(APP_TITLE)
logger.setLevel(logging.DEBUG)


# load FaceNet model
weight_path = "facenet_weights.h5"
model = load_model(weight_path)

##### Run model inferrence first time after starting api service #####
def test_inferrence():
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
logger.info("Running inferrence testing")
test_inferrence()
######################################################################


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)


@app.get("/")
def read_root():
    return """Face Verification API. 
    Visit `/docs` to use api swagger.
    """


@app.post("/verify/withimage")
async def verify_face_with_image(
    imgFileToVerify: UploadFile,
    imgFileAuthentic: UploadFile,
) -> Verification:
    # Read images
    if VERBOSE:
        logger.info("Reading images")
    if not imgFileToVerify.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be jpg or png format.",
        )
    if not imgFileAuthentic.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be jpg or png format.",
        )
    img_to_verify = read_img_file(await imgFileToVerify.read())
    img_authentic = read_img_file(await imgFileAuthentic.read())

    # Detect face in a to-verify image
    if VERBOSE:
        logger.info("Detecting face in a to-verify image")
    faces_to_verify = face_verification.detect(
        img_to_verify,
        target_size=model.input_shape[1:3],
    )
    if len(faces_to_verify) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face could not be detected in a to-verify image.",
        )
    elif len(faces_to_verify) > 1:
        logger.warning(
            f"{len(faces_to_verify)} faces have been detected in a to-verify image. "
            + "Only one with the highest confidence is used for verification."
        )

    # Detect face in an authentic image
    if VERBOSE:
        logger.info("Detecting face in an authentic image")
    faces_authentic = face_verification.detect(
        img_authentic,
        target_size=model.input_shape[1:3],
    )
    if len(faces_authentic) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face could not be detected in an authentic image.",
        )
    elif len(faces_authentic) > 1:
        logger.warning(
            f"{len(faces_authentic)} faces have been detected in an authentic image. "
            + "Only one with the highest confidence is used for verification."
        )

    # Embed face images to feature vectors
    if VERBOSE:
        logger.info("Embedding face images")
    embedding_to_verify = face_verification.embed(
        faces_to_verify[0]["face"],
        embedding_model=model,
    )
    embedding_authentic = face_verification.embed(
        faces_authentic[0]["face"],
        embedding_model=model,
    )

    # Calculate similarity between two embeddings
    if VERBOSE:
        logger.info("Calculating similarity")
    confidence = face_verification.calculate_similarity(
        embedding_to_verify,
        embedding_authentic,
    )

    if VERBOSE:
        logger.info("Returning response")
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


@app.post("/detect")
async def detect_faces(
    imgFile: UploadFile,
)->List[FaceObj]:
    if VERBOSE:
        logger.info("Reading image")
    if not imgFile.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be jpg or png format.",
        )
    img = read_img_file(await imgFile.read())

    if VERBOSE:
        logger.info("Detecting faces in an image")
    faces = face_verification.detect(
        img,
        target_size=model.input_shape[1:3],
    )
    if len(faces) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face could not be detected in an image.",
        )
    
    faces_ = [
        dict(
            detectionConfidence=face_obj["confidence"],
            area=face_obj["facial_area"],
        )
        for face_obj in faces
    ]
    return faces_


@app.post("/detectandembed")
async def detect_and_embed_face(
    imgFile: UploadFile,
):
    if VERBOSE:
        logger.info("Reading image")
    if not imgFile.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be jpg or png format.",
        )
    img = read_img_file(await imgFile.read())

    if VERBOSE:
        logger.info("Detecting faces in an image")
    faces = face_verification.detect(
        img,
        target_size=model.input_shape[1:3],
    )
    if len(faces) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face could not be detected in an image.",
        )
    elif len(faces) > 1:
        logger.warning(
            f"{len(faces)} faces have been detected in an image. "
            + "Only one with the highest confidence would be embedded."
        )

    if VERBOSE:
        logger.info("Embedding a face image")
    embedding = face_verification.embed(
        faces[0]["face"],
        embedding_model=model,
    )

    response = json.dumps(embedding.tolist()) # convert numpy array to list
    return response


if __name__ == "__main__":
    uvicorn.run(app)
