# keras-flask-app

A single page web app which recognizes drawn digits using a machine learning model.

It uses a CNN model constructed and trained using Keras with Tensorflow as backend. Digit classification using model happens at server-side and results are returned to client in form of JSON data.

## Built Using
* Flask (Python) for back-end
* Good ol' Vanilla JS for front-end

## Setup
For this you must have Python3 and pip3 installed and added to PATH environment variable, and they must be your default Python and pip. You should also use a virtual environment tool like [virtualenv](https://virtualenv.pypa.io/en/stable/) or [conda](https://conda.io/docs/).
* Clone or download the project repo.
* Open command line (linux terminal/windows cmd) and cd to project directory.
* If you use virualenv or conda, create a virtual environment and activate it.
* Run following in command line:
```
pip install -n requirments.txt
```
* Then run:
```
python run.py
```
* Wait till you see the message (or any other similar one):
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 ```
 * By default Flask server listens to port 5000 of localhost. Open [localhost:5000](http://localhost:5000) in your browser.

 Now you can play with the app. Draw any digit from 0 to 9 in rectangle, click send and see magic of ML.
 
 Create an issue on my repo if you run into any error while doing this.