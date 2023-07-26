import numpy as np
from pydantic import BaseModel
from dataclasses import dataclass


class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int


@dataclass
class FaceObj:
    face: np.ndarray
    facial_area: FacialArea
    confidence: float


class DetectionObj(BaseModel):
    detectionConfidence: float
    area: FacialArea


class RecognitionObj(BaseModel):
    subject_id: str
    confidence: float


class VerificationObj(BaseModel):
    confidence: float
    faceToVerify: DetectionObj
    faceAuthentic: DetectionObj
