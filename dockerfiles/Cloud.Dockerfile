# Docker Image to run on Cloud (Vultr NGC)

FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

WORKDIR /code

RUN pip uninstall typing-extensions -y
RUN pip install fastapi==0.83.0
RUN pip install uvicorn==0.16.0
RUN pip install opencv-python==4.7.0.72
RUN pip install mtcnn==0.1.1
RUN pip install matplotlib==3.3.4
RUN pip install pandas==1.3.3
RUN pip install gdown
RUN pip install deepface==0.0.79

RUN apt-get update && apt-get install libgl1 -y

RUN pip install -U --force-reinstall typing-extensions
RUN pip install python-multipart

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log.ini"]
