FROM tensorflow/tensorflow:latest-gpu
WORKDIR /var/task
RUN python -m pip install matplotlib
RUN python -m pip install scipy
