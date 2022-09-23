import json
#import pandas
import numpy as np
import joblib
import boto3

ENDPOINT_NAME = "sagemaker-scikit-learn-2022-06-23-22-49-48-547"
runtime= boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    #1. Parse out query string params

      
    # my_pickle = joblib.load(s3.Bucket(S3_BUCKET_NAME).Object("Save Model\OHE.pkl").get()['Body'].read())
    
    
    price = 5
    print(json.loads(json.dumps(event)))
    
    if json.loads(json.dumps(event))['body']:
        body = json.loads(json.loads(json.dumps(event))['body'])
        print(body)
        ofertaOd = body['ofertaOd']
        rokProdukcji = body['rokProdukcji']
        przebieg = body['przebieg']
        pojemnoscSkokowa = body['pojemnoscSkokowa']
        moc = body['moc']
        price = rokProdukcji
        
        dct = {'data': [[rokProdukcji, przebieg, pojemnoscSkokowa, przebieg]]}
        payload = dct['data']
        print(payload)
        
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                           Body=json.dumps(payload))
        result = json.loads(response['Body'].read().decode())
        
        price = result[0]
        print("price", price)
        price = price - 0.2*przebieg
        price = price + 100*(rokProdukcji-2020)
        price = price + 10*pojemnoscSkokowa
        price = int(price)
    responseObject = {}
    responseObject['statusCode'] = 200
    responseObject['headers'] = {}
    responseObject['headers']['Content-Type'] = 'application/json'
    responseObject['headers']['Access-Control-Allow-Origin'] = '*'
    responseObject['headers']['Access-Control-Allow-Methods'] = '*'
    responseObject['headers']['Access-Control-Allow-Headers'] = '*'
    responseObject['body'] = json.dumps(price)
    return responseObject