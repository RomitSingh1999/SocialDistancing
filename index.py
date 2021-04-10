from flask import Flask, render_template, url_for, redirect, request,Response
from pytool import social_distancing_config as config
from pytool.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from PIL import Image
from camera import VideoCamera
app = Flask(__name__)


@app.route('/')
def hello_world():
    return "Hello World"


@app.route('/about')
def index():
     
    return render_template("index.html")

def videorender():
    cap = cv2.VideoCapture(0)


    imgpath = './static/cap.jpeg'
    while(True): 
        ret, frame = cap.read()
         
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(imgpath, frame)
        videorender(imgpath)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
