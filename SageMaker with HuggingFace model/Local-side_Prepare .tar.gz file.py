from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Remember to install the HuggingFace version that match with the version when deployed on AWS

# Load from HF
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Save to my_model
tokenizer = AutoTokenizer.from_pretrained("./my_model/my_tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("./my_model/my_model")

content = """
from sagemaker_huggingface_inference_toolkit import decoder_encoder
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

tokenizer = None

def model_fn(model_dir):
    # implement custom code to load the model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "my_tokenizer"))
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_dir, "my_model"))
    return loaded_model 

def input_fn(input_data, content_type):
    # decode the input data  (e.g. JSON string -> dict)
    input_data = json.loads(input_data)
    data = tokenizer.encode(input_data['my_input'], return_tensors = 'pt')
    if content_type == "OK":
      print("OK-content_type")
      return tokenizer.encode("", return_tensors = 'pt')
    return data

def predict_fn(data, model):
    # call your custom model with the data
    prediction = model.generate(data, max_length=150, min_length=100)
    return prediction

def output_fn(prediction, accept):
    # convert the model output to the desired output format (e.g. dict -> JSON string)
    response = tokenizer.decode(prediction[0])
    return {"output": response}"""

os.mkdir('my_model/code')

open('my_model/code/inference.py', 'w').write(content)
open('my_model/code/requirements.txt', 'w').write("transformers[sentencepiece]\ntransformers==4.26")

# Generate the .tar.gz file
os.system('!tar -czvf my_model.tar.gz -C my_model .')

# Upload file model to Amazon S3
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id="<aws_access_key_id>",
    aws_secret_access_key= "<aws_secret_access_key>"
    )

s3.upload_file("./my_model.tar.gz", "sang-sagemaker-practice", "my_model.tar.gz")

# Call an incoke to Endpoint
import boto3, json, sagemaker
import os 

os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

sagemaker_session = sagemaker.Session(boto3.session.Session(
    aws_access_key_id="<aws_access_key_id>",
    aws_secret_access_key= "<aws_secret_access_key>"
))

predictor = sagemaker.base_predictor.Predictor(
    endpoint_name = 'huggingface-pytorch-inference-2024-01-13-02-53-41-283', 
    sagemaker_session = sagemaker_session,
    serializer = sagemaker.base_serializers.JSONSerializer(),
    deserializer = sagemaker.base_deserializers.JSONDeserializer()
)

response=predictor.predict(data)
response
