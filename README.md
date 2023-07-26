# Face API

Face API, api for Face Detection, Face Verification, and Face Recognition. It is built with FastAPI. Available detection model in api includes MTCNN and OpenCV's Haar Cascade. FaceNet is used as embedding model.

# Run API

Use following command to start api. [(read more details)](https://fastapi.tiangolo.com/deployment/manually/)
`uvicorn main:app --port 8000 --log-config log.ini`
