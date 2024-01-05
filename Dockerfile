ARG IMAGE_NAME
FROM ${IMAGE_NAME}:12.3.1-base-ubuntu22.04 as base

ENV NV_CUDA_LIB_VERSION 12.3.1-1

FROM base as base-amd64

ENV NV_NVTX_VERSION 12.3.101-1
ENV NV_LIBNPP_VERSION 12.2.3.2-1
ENV NV_LIBNPP_PACKAGE libnpp-12-3=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.2.0.103-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-3
ENV NV_LIBCUBLAS_VERSION 12.3.4.1-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.19.3-1
ENV NCCL_VERSION 2.19.3-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.3

FROM base as base-arm64

ENV NV_NVTX_VERSION 12.3.101-1
ENV NV_LIBNPP_VERSION 12.2.3.2-1
ENV NV_LIBNPP_PACKAGE libnpp-12-3=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.2.0.103-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-3
ENV NV_LIBCUBLAS_VERSION 12.3.4.1-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.19.3-1
ENV NCCL_VERSION 2.19.3-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.3

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-3=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-3=${NV_NVTX_VERSION} \
    libcusparse-12-3=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# Add entrypoint items
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# Install any other dependencies and packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "./tsnejax.py"]
