from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # import dataset
    df1 = pd.read_excel('dataset/fake_new_dataset1.xlsx')

    # Extract Feature With TF-IDF model 
    label = df1.iloc[:, -1] 
    text = df1.iloc[:, 1]                               
    vectorizer = TfidfVectorizer(max_features= 5000)
    vectorizer.fit_transform(text).toarray()


    #Load the Model
    model = open("model/SVM.pkl","rb")                         
    model = pickle.load(model)                                 

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = vectorizer.transform(data).toarray()
        my_prediction = model.predict(vect)
        return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)