FROM python:3.10-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src/app
RUN pip install poetry==1.8.2
COPY commands.py poetry.lock pyproject.toml /usr/src/app
COPY pos_detection /usr/src/app/pos_detection
COPY test_imgs /usr/src/app/test_imgs
RUN poetry install
