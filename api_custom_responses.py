from pydantic import BaseModel


API_STATUS_CODE = {
    "NOT_SUPPORTED_IMAGE_FILE": 480,
    "FACE_NOT_DETECTED": 481,
}

API_STATUS_DETAIL = {
    "NOT_SUPPORTED_IMAGE_FILE": "Image file must be jpg or png format.",
    "FACE_NOT_DETECTED": "Face could not be detected in an image.",
}

API_STATUS_DESCRIPTION = {
    "NOT_SUPPORTED_IMAGE_FILE": "Not Supported Image File Error",
    "FACE_NOT_DETECTED": "Face Not Detected Error",
}


class HTTPError(BaseModel):
    detail: str


# STATUS 480
class HTTPNotSupportImageFileError(HTTPError):
    detail = "Image file must be jpg or png format."


# STATUS 481
class HTTPFaceNotDetectedError(HTTPError):
    detail = "Face could not be detected in an image."


API_RESPONSES = {
    API_STATUS_CODE["NOT_SUPPORTED_IMAGE_FILE"]: {
        "description": API_STATUS_DESCRIPTION["NOT_SUPPORTED_IMAGE_FILE"],
        "model": HTTPNotSupportImageFileError,
    },
    API_STATUS_CODE["FACE_NOT_DETECTED"]: {
        "description": API_STATUS_DESCRIPTION["FACE_NOT_DETECTED"],
        "model": HTTPFaceNotDetectedError,
    },
}
