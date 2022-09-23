import os
import io
import re
import json
import logging
import boto3
import urllib.parse
import re
from botocore.exceptions import ClientError
from hate_speech-detection import one_hot_encode
from hate_speech-detection import vectorize_sequences

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

def lambda_handler(event, context):
    
    logger.info('printing event')
    logger.info(event)
    
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    print("bucket  ", bucket)
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print("key  ", key)
    data = s3.get_object(Bucket=bucket,Key= key)
    contents = data['Body'].read()
    speech = message_from_bytes(contents)
    

    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    runtime= boto3.client('runtime.sagemaker')   
    
    payload = ""
    
    if speech.is_multipart():
        print("multi part")
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

        # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                payload = part.get_payload(decode=True)  # decode
                print("multi part", payload)
                break
    else:
        #print("msg payload is = ", msg.get_payload())
        payload = msg.get_payload()
        
    
    print("payload is ", payload.decode("utf-8"))
    payload = payload.decode("utf-8")
    #re.sub('\s', " " , payload)
    payload = payload.replace('\r\n',' ').strip()
    
    payloadtext = payload
    
    vocabulary_length = 9013
    hate_speech = [payload]
    #one_hot_hate_speech = ["I support racism. I don't care"]
    one_hot_hate_speech = one_hot_encode(test_messages, vocabulary_length)
    encoded_hate_speech = vectorize_sequences(one_hot_hate_speech, vocabulary_length)
    payload = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=payload)
    
    response_body = response['Body'].read().decode('utf-8')
    result = json.loads(response_body)
    print(result)
    pred = int(result['predicted_label'][0][0])
    if model.predict(vect) == 1:
       print("Hate Speech")
    else:
      print("Free Speech")
    
    