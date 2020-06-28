import argparse
import logging
import sagemaker_containers
import requests
from subprocess import call

import os
import io
import glob
import time

from fastai.vision import *
import fastai

# def upgrade():
#     call(['pip', 'install', 'fastai'])

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


def model_fn(model_dir):
    logger.info('model_fn')
    path = Path(model_dir)
    print('Creating DataBunch object')
    empty_data = ImageDataBunch.load_empty(path)
    arch_name = os.path.splitext(os.path.split(glob.glob(f'{model_dir}/resnet*.pth')[0])[1])[0]
    print(f'Model architecture is: {arch_name}')
    arch = getattr(models, arch_name)    
    learn = create_cnn(empty_data, arch, pretrained=False).load(path/f'{arch_name}')
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE:
        img = open_image(io.BytesIO(request_body))
        return img
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        img_request = requests.get(request_body['url'], stream=True)
        img = open_image(io.BytesIO(img_request.content))
        return img        
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    response = {}
    response['class'] = str(predict_class)
    response['confidence'] = predict_values[predict_idx.item()].item()
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        output = json.dumps(prediction)
        return output, accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    


print("Inside Train.py")