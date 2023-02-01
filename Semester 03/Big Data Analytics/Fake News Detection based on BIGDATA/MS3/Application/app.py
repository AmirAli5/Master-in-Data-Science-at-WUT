from flask import Flask,render_template,url_for,request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # import TF-IIDF Vectorizer
    tf_idf = open("model/cv.pkl","rb")           
    cv = pickle.load(tf_idf)    
            
    # import pickle file of my model
    model = open("model/model.pkl","rb")
    clf = pickle.load(model)
    
    if request.method == 'POST':
    	comment = request.form['comment']
    	data = [comment]
    	vect = cv.transform(data).toarray()
    	my_prediction = clf.predict(vect)
    	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)