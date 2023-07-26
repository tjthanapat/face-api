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

RUN pip3 install mtcnn==0.1.1 --no-deps


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log.ini"]
