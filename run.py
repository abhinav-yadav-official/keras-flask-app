"""
run.py
"""

from flask import Flask
from flask import request,render_template,jsonify

import image
import classifier

app = Flask(__name__)

@app.route('/')
def page():
    return render_template('page.html')

@app.route('/image',methods=['POST'])
def image_post_request():
    x = image.convert(request.json['image'])
    y,n = classifier.classify(x)
    return jsonify({'result':y,'digit':n})

if __name__ == '__main__':
	app.run()