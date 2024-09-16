FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --set python3 /usr/bin/python3.11

# Install pip for Python 3.11
RUN apt-get update && apt-get install -y curl \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && apt-get remove -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y jq
RUN mkdir model raw_data processed_data results

COPY ../../raw_data/sentimientos.xlsx ./raw_data/sentimientos.xlsx
COPY ../../helpers.py ./helpers.py
COPY ../../preprocessing.py ./preprocessing.py
COPY ../../train.py ./train.py
COPY ../../test.py ./test.py

COPY ../../requirements_train_test.txt ./requirements_train_test.txt
RUN pip install -r requirements_train_test.txt