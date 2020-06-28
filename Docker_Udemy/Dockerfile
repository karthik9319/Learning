#working
# FROM python:3.7-slim
# LABEL maintainer = "Karthik Vullam"

# ADD train.py /


# RUN apt-get update && \
#     apt-get -y install gcc mono-mcs && \
#     rm -rf /var/lib/apt/lists/* 

# RUN pip install pip==18.1
# RUN pip install --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir flask
# RUN pip install --no-cache-dir boto3
# RUN pip install --no-cache-dir pillow
# RUN pip install --no-cache-dir gunicorn
# RUN pip install torch torchvision
# RUN pip install sagemaker_containers
# RUN pip install --no-cache-dir fastai

# EXPOSE 5000

# CMD ["python3", "./train.py"]

FROM python:3.7-slim
LABEL maintainer = "Karthik Vullam"

# ADD serve.py /
# ENV /serve.py 
# ADD /serve $qpath
# COPY /serve.py $(PWD)/app/serve.py
# RUN echo "$PWD"
COPY serve.py .

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
RUN pip install fastai
RUN pip install sagemaker_containers


# WORKDIR /app

CMD [ "python", "serve.py" ]