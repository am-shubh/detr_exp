FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev ffmpeg libsm6 libxext6 && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

RUN pip install -q supervision
RUN pip install -q transformers
RUN pip install -q pytorch-lightning
RUN pip install -q timm

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

COPY . /workspace/

WORKDIR /workspace

CMD python main.py