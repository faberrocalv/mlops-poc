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

COPY ./requirements_fastapi.txt ./requirements_fastapi.txt

RUN pip install -r requirements_fastapi.txt

COPY ./helpers.py ./helpers.py
COPY ./main.py ./main.py
COPY ./predict.py ./predict.py

CMD ["fastapi", "run", "main.py", "--port", "8000"]