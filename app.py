from flask import Flask, render_template, request, url_for
import joblib
from process import process_text as pt
import __main__
__main__.process_text = pt
app = Flask(__name__)

@app.before_request
def load_model():
    global model
    model= joblib.load(open('fakenewssvm.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/artikel')
def artikel():
    return render_template('artikel.html')



@app.route('/result', methods=['POST'])
def result():
    message = request.form['message']
    data = [message]
    pre = model.predict(data)
    prob = model.predict_proba(data)
    prob_fake = round((prob[0][1]*100),2)
    
    prob_true = round((prob[0][0]*100),2)
    hasil= "Sekian"
    return render_template('result.html',prediction=pre[0],text=message, prob_fake=prob_fake, prob_true= prob_true)

if __name__ == '__main__':
    app.run(threaded=True, debug=True)