# Docker Image to run on Jetson Nano

FROM l4t-ml:tf2.4.0

WORKDIR /code

RUN pip3 uninstall typing-extensions -y
RUN pip3 install fastapi==0.83.0
RUN pip3 install uvicorn==0.16.0
RUN pip3 install python-multipart
RUN pip3 install gdown
RUN pip3 install --no-deps deepface

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip3 install mtcnn==0.1.1


# COPY ./facenet_weights.h5 /code/facenet_weights.h5
# COPY ./facenet.py /code/facenet.py
# COPY ./test_image.png /code/test_image.png
# COPY ./face_verification.py /code/face_verification.py
# COPY ./utils.py /code/utils.py
# COPY ./main.py /code/main.py


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log.ini"]
