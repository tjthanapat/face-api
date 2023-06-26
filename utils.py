from io import BytesIO
from PIL import Image
import numpy as np
from pydantic import BaseModel


class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceObj(BaseModel):
    detectionConfidence: float
    area: FacialArea


class VerificationResponse(BaseModel):
    confidence: float
    faceToVerify: FaceObj
    faceAuthentic: FaceObj


def read_img_file(file) -> np.ndarray:
    img = np.array(Image.open(BytesIO(file)))
    return img
