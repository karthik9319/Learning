FROM python:3.7-slim
LABEL maintainer = "Karthik Vullam"

ADD train.py /
# WORKDIR /app

RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/* 

RUN pip install pip==18.1
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir flask
RUN pip install --no-cache-dir boto3
RUN pip install --no-cache-dir pillow
RUN pip install --no-cache-dir gunicorn
RUN pip install torch torchvision
# RUN pip install sagemaker_containers
# RUN pip install --no-cache-dir torch
# RUN pip install --no-cache-dir torchvision
RUN pip install --no-cache-dir fastai

# ADD . /app
# COPY . .
EXPOSE 5000

# ENTRYPOINT ["python3", "train.py" ]
CMD ["python3", "./train.py"]