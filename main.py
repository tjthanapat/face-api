import tensorflow as tf

tf.keras.utils.disable_interactive_logging()  # Turn off interactive logging
# tf.config.set_visible_devices([], 'GPU') # Hide GPU from visible devices


import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, status

import face_verification
from facenet import load_model
from utils import read_img_file, VerificationResponse


VERBOSE = True  # Flag for printing api logging
if VERBOSE:
    import logging

    logger = logging.getLogger("Face Verification API")
    logger.setLevel(logging.DEBUG)


APP_TITLE = "Face Verification API"
APP_VERSION = "1.0.1"
APP_DESCRIPTION = """
Face Verification API.
"""

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)


@app.get("/")
def read_root():
    return dict(message="Face Verification API. Visit `/docs` to use api swagger.")


# load FaceNet model
weight_path = "facenet_weights.h5"
model = load_model(weight_path)


@app.post("/test_upload_file")
async def test_upload_file(file: UploadFile):
    if not file.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        return "Image must be jpg or png format."
    else:
        return "pass"


@app.post("/verify/withimage")
async def verify_face_with_image(
    imgFileToVerify: UploadFile,
    imgFileAuthentic: UploadFile,
) -> VerificationResponse:
    # Read images
    if VERBOSE:
        logger.info("Reading images")
    if not imgFileToVerify.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be jpg or png format.",
        )
    if not imgFileAuthentic.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be jpg or png format.",
        )
    img_to_verify = read_img_file(await imgFileToVerify.read())
    img_authentic = read_img_file(await imgFileAuthentic.read())

    # Detect face in to-verify image
    if VERBOSE:
        logger.info("Detecting face in to-verify image")
    faces_to_verify = face_verification.detect(
        img_to_verify,
        target_size=model.input_shape[1:3],
    )
    # Detect face in authentic image
    if VERBOSE:
        logger.info("Detecting face in authentic image")
    faces_authentic = face_verification.detect(
        img_authentic,
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


if __name__ == "__main__":
    uvicorn.run(app)
