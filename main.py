#!/usr/bin/env python

from flask import Flask
from flask import render_template, jsonify, Response
from PIL import Image
from gan_sampler import Generate

app = Flask(__name__)

def cut():
    img = Image.open("sample.png")
    img2 = img.crop((0, 0, 128, 128))
    img2.save("target.jpg")
    return 0

@app.route("/")
def index():
    Generate()
    cut()
    with open('target.jpg', 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image/jpeg")
        return resp


app.run(debug=True)
