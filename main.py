import warnings

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.exceptions import ValidationError
from fastapi.responses import JSONResponse

import face_verification
from facenet import load_model

from utils import read_img_file, VerificationResponse

import tensorflow as tf

# tf.config.set_visible_devices([], 'GPU') # Hide GPU from visible devices


description = """
Face Verification API.
"""

app = FastAPI(
    title="Face Verification API",
    version="1.0.0",
    description=description,
)


@app.get("/")
def read_root():
    return dict(message="Face Verification API. Visit `/docs` to use api swagger.")


# @app.exception_handler(ValidationError)
# async def value_error_exception_handler(request: Request, exc: ValidationError):
#     return JSONResponse(
#         status_code=status.HTTP_400_BAD_REQUEST,
#         content={"message": str(exc)},
#     )


# load FaceNet model
weight_path = "facenet_weights.h5"
model = load_model(weight_path)


@app.post("/test_upload_file")
async def test_upload_file(file: UploadFile = File(...)):
    if not file.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        return "Image must be jpg or png format."
    else:
        return "pass"


@app.post("/verify/withimage")
async def verify_face_with_image(
    imgFileToVerify: UploadFile = File(...),
    imgFileAuthentic: UploadFile = File(...),
) -> VerificationResponse:
    # Read images
    if not imgFileToVerify.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        return "Image must be jpg or png format."
    if not imgFileAuthentic.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        return "Image must be jpg or png format."
    img_to_verify = read_img_file(await imgFileToVerify.read())
    img_authentic = read_img_file(await imgFileAuthentic.read())

    # Detect face in to-verify image
    faces_to_verify = face_verification.detect(
        img_to_verify,
        target_size=model.input_shape[1:3],
    )
    if len(faces_to_verify) == 0:
        return "Face could not be detected in to-verify image."
    elif len(faces_to_verify) > 1:
        warnings.warn(
            f"{len(faces_to_verify)} faces have been detected in to-verify image. "
            + "Only the first one is used for verification."
        )

    # Detect face in authentic image
    faces_authentic = face_verification.detect(
        img_authentic,
        target_size=model.input_shape[1:3],
    )
    if len(faces_authentic) == 0:
        return "Face could not be detected in authentic image."
    elif len(faces_authentic) > 1:
        warnings.warn(
            f"{len(faces_authentic)} faces have been detected in authentic image. "
            + "Only the first one is used for verification."
        )

    # Embed face images to feature vectors
    embedding_to_verify = face_verification.embed(
        faces_to_verify[0]["face"],
        embedding_model=model,
    )
    embedding_authentic = face_verification.embed(
        faces_authentic[0]["face"],
        embedding_model=model,
    )

    # Calculate similarity between two embeddings
    confidence = face_verification.calculate_similarity(
        embedding_to_verify,
        embedding_authentic,
    )

    result = dict(
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

    return result


if __name__ == "__main__":
    uvicorn.run(app)
