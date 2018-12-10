#!/usr/bin/env python

from flask import Flask
from flask import render_template

from flask import jsonify
from flask import Response
app = Flask(__name__)

@app.route("/")
def index():
    with open('sample.png', 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image/jpeg")
        return resp


app.run(debug=True)
