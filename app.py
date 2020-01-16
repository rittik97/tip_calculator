from flask import render_template
import flask
import pickle
import pandas as pd
import numpy as np
import json

app = flask.Flask(__name__, template_folder='templates')

#-------- MODEL GOES HERE -----------#


pipe = pickle.load(open("pipe.pkl", 'rb'))



#-------- ROUTES GO HERE -----------#

@app.route('/')
def index():
    with open("templates/index.html", 'r') as p:
       return p.read()
@app.route('/hist', methods=['POST'])
def trial():
        args= (flask.request.form)
        args = dict(flask.request.form)
        #print(args)
        data=pd.DataFrame({
            'total_bill': [float(args.get('data[bill]'))],
            'sex': args.get('data[sex]'),
            'smoker': args.get('data[smoker]'),
            'day': args.get('data[day]'),
            'time': args.get('data[time]'),
            'size': [float(args.get('data[guests]'))]

        })
        #print(data)
        pred=np.round(pipe.predict(data)[0],2)

        return json.dumps(pred)

@app.route('/about')
def about():
        return render_template('about.html')



if __name__ == '__main__':
    app.run(port=5000, debug=True)
