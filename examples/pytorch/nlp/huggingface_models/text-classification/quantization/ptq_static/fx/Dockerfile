#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################
# Build the basic Env (Not include Neural-Compressor)
# https://github.com/intel/neural-compressor/blob/master/.azure-pipelines/docker/Dockerfile.devel
###########################################################

ARG UBUNTU_VER=20.04
FROM ubuntu:${UBUNTU_VER}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    autoconf \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    numactl \
    time \
    wget \
    bc \
    vim

RUN ln -sf $(which python3) /usr/bin/python

RUN python -m pip  install --upgrade pip
RUN python -m pip install setuptools

RUN pip list

    
    
###########################################################
# Allow Pods communicate through OpenSSH wihout need confirmation
# https://github.com/kubeflow/mpi-operator/blob/master/examples/v2beta1/horovod/Dockerfile
###########################################################

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd


# Allow OpenSSH to talk to containers without asking for confirmation
# by disabling StrictHostKeyChecking.
# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# to disable UserKnownHostsFile to avoid write permissions.
# Disabling StrictModes avoids directory and files read permission checks.
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config



###########################################################
# Install requirements for tasks
###########################################################

###########################################################
# Build OpenMPI
# https://github.com/kubeflow/mpi-operator/blob/master/examples/v2beta1/horovod/Dockerfile
###########################################################

# Install Open MPI
# TODO should use the 4.1?
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.mpich.org/static/downloads/4.1.2/mpich-4.1.2.tar.gz && \
    tar zxf mpich-4.1.2.tar.gz && \
    cd mpich-4.1.2 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi
# RUN apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev

RUN echo "check mpicc ..."
RUN which mpicc
RUN which mpirun



###########################################################
# Create a workdir and add task files
###########################################################

RUN apt-get update && \
    apt-get install -y ncdu

CMD ncdu /

RUN cd /home  && \
    git clone https://github.com/intel/neural-compressor.git INC && \
    cd INC && \
    cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx  && \
    pip install neural-compressor && \
    pip install -r requirements.txt && \
    pip install mpi4py


RUN cd /var && \
    git clone https://github.com/intel/neural-compressor.git inc && \
    cd inc && \
    git checkout -b k8s origin/ly/k8s && \
    ls


