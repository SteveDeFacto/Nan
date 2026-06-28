# NAN GPU worker. Needs the CUDA toolkit (PyCUDA compiles against it) + curl for
# the healthcheck. Config gives it runtime:nvidia, gpus:all, ipc:host, and the
# shared /tmp/nvidia-mps mount so it joins the MPS daemon.
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir pycuda==2025.1

COPY worker.py /worker.py
ENV WORKER_PORT=8090
EXPOSE 8090
CMD ["python3", "/worker.py"]
