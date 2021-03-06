# parameters
ARG REPO_NAME="dt-machine-learning-base-environment"
ARG MAINTAINER="Andrea F. Daniele (afdaniele@ttic.edu)"
ARG DESCRIPTION="Base image containing common libraries and environment setup for Machine Learning applications."
ARG ICON="square"

ARG ARCH=amd64
ARG DISTRO=daffy
ARG BASE_TAG=${DISTRO}-${ARCH}
ARG BASE_IMAGE=dt-commons
ARG LAUNCHER=default



# define base image
FROM duckietown/${BASE_IMAGE}:${BASE_TAG} as BASE

# recall all arguments
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG ICON
ARG ARCH
ARG DISTRO
ARG ROS_DISTRO
ARG BASE_TAG
ARG BASE_IMAGE
ARG LAUNCHER

# nvidia environment
ENV PATH /usr/local/cuda-10.1/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
ENV LANG C.UTF-8

# define and create repository path
ARG REPO_PATH="${SOURCE_DIR}/src/${REPO_NAME}"
ARG LAUNCH_PATH="${LAUNCH_DIR}/${REPO_NAME}"
RUN mkdir -p "${REPO_PATH}"
RUN mkdir -p "${LAUNCH_PATH}"
WORKDIR "${REPO_PATH}"

# keep some arguments as environment variables
ENV DT_MODULE_TYPE "${REPO_NAME}"
ENV DT_MODULE_DESCRIPTION "${DESCRIPTION}"
ENV DT_MODULE_ICON "${ICON}"
ENV DT_MAINTAINER "${MAINTAINER}"
ENV DT_REPO_PATH "${REPO_PATH}"
ENV DT_LAUNCH_PATH "${LAUNCH_PATH}"
ENV DT_LAUNCHER "${LAUNCHER}"

# create repo directory
RUN mkdir -p "${REPO_PATH}"

# setup Nvidia repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

# install CUDA, CUDNN
COPY ./dependencies-apt.txt "${REPO_PATH}/"
RUN dt-apt-install "${REPO_PATH}/dependencies-apt.txt" \
    && apt-mark hold \
        libnccl2 \
        libcublas10 \
        libcudnn7 \
        libnccl-dev \
        libcublas-dev\
    && apt-get autoremove -y  \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s cuda-10.1 /usr/local/cuda \
    && ln -s $(which ${PYTHON}) /usr/local/bin/python \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf


# install python dependencies
COPY ./dependencies-py3.txt "${REPO_PATH}/"
RUN pip3 install --use-feature=2020-resolver -r ${REPO_PATH}/dependencies-py3.txt -f https://download.pytorch.org/whl/torch_stable.html

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
COPY ./requirements.txt "${REPO_PATH}/"
RUN pip3 install --use-feature=2020-resolver -r ${REPO_PATH}/requirements.txt

# copy the source code
COPY ./packages "${REPO_PATH}/packages"

# install launcher scripts
COPY ./launchers/. "${LAUNCH_PATH}/"
RUN dt-install-launchers "${LAUNCH_PATH}"

RUN pip3 uninstall -y dataclasses

# define default command
CMD ["bash", "-c", "dt-launcher-${DT_LAUNCHER}"]

# store module metadata
LABEL org.duckietown.label.module.type="${REPO_NAME}" \
    org.duckietown.label.module.description="${DESCRIPTION}" \
    org.duckietown.label.module.icon="${ICON}" \
    org.duckietown.label.architecture="${ARCH}" \
    org.duckietown.label.code.location="${REPO_PATH}" \
    org.duckietown.label.code.version.distro="${DISTRO}" \
    org.duckietown.label.base.image="${BASE_IMAGE}" \
    org.duckietown.label.base.tag="${BASE_TAG}" \
    org.duckietown.label.maintainer="${MAINTAINER}"

# nvidia labels
LABEL com.nvidia.cudnn.version="7.6.5.32"
WORKDIR /submission
