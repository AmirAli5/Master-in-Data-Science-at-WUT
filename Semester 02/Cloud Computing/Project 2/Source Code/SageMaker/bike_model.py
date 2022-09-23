from __future__ import print_function

import argparse
import os
import pandas as pd
import sklearn
import numpy as np

from sklearn import tree
from sklearn.externals import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "scrapped_23052022.csv")
    df = pd.read_csv(file, engine="python")
    
    # Predictor Attributes
    X = df.iloc[:, 5:9]
    Y = df.iloc[:, 21]
    
    
    X['Przebieg'] = X['Przebieg'].fillna(X['Przebieg'].mode()[0])
    X['Pojemność skokowa'] = X['Pojemność skokowa'].fillna(df['Pojemność skokowa'].mode()[0])
    X['Moc'] = X['Moc'].fillna(X['Moc'].mode()[0])
  
    
    X['Przebieg'] = X['Przebieg'].str.extract(r'(\d+)').astype(int)
    X['Pojemność skokowa'] = X['Pojemność skokowa'].str.extract(r'(\d+)').astype(int)
    X['Moc'] = X['Moc'].str.extract(r'(\d+)').astype(int)
    
    
    # Encoding categorical data
    #from sklearn.preprocessing import OneHotEncoder
    #onehotencoder = OneHotEncoder(sparse=False,)
    #X = onehotencoder.fit_transform(X)
    Y = df.iloc[:, 21]
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(regressor, os.path.join(args.model_dir, "model.joblib"))
    
    # print the onehot encoder and save
    #joblib.dump(onehotencoder, os.path.join(args.ohe_dir, "ohe.joblib"))

    
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    regressor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regressor


#def predict_fn(input_data, model):
#    prediction = model.predict(input_data)
#    pred_prob = model.predict_proba(input_data)
#    return np.array([prediction, pred_prob])