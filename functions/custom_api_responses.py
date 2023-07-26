from pydantic import BaseModel


API_STATUS_CODE = {
    "NOT_SUPPORTED_IMAGE_FILE": 480,
    "FACE_NOT_DETECTED": 481,
    "NO_DB_FILE_FOUND": 482,
    "NO_EMBEDDING_IN_DB": 483,
    "NO_SUBJECT_ID_IN_DB": 484,
}


class HTTPMessageError(BaseModel):
    detail: str


API_RESPONSES = {
    "NOT_SUPPORTED_IMAGE_FILE": {
        "description": "Not Supported Image File Error",
        "model": HTTPMessageError,
    },
    "FACE_NOT_DETECTED": {
        "description": "Face Not Detected Error",
        "model": HTTPMessageError,
    },
    "NO_DB_FILE_FOUND": {
        "description": "No DB File Found Error",
        "model": HTTPMessageError,
    },
    "NO_EMBEDDING_IN_DB": {
        "description": "No Embedding in DB Error",
        "model": HTTPMessageError,
    },
    "NO_SUBJECT_ID_IN_DB": {
        "description": "No Embedding with a Given Subject ID in DB Error",
        "model": HTTPMessageError,
    },
}
