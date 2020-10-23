import views
from flask import Flask, Response
import cv2
from camera import VideoCamera
from flask import render_template, request
import time
from mtcnn import MTCNN
from PIL import Image, ImageDraw
import os
import numpy as np
from flask_sqlalchemy import SQLAlchemy 
from faces import facenett

app = Flask(__name__)
video = cv2.VideoCapture(0)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app.add_url_rule('/base','base',views.base)
app.add_url_rule('/login','login',views.login)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/faceapp','faceapp',views.faceapp)
app.add_url_rule('/gender','gender',views.gender,methods=['GET','POST'])

@app.route('/video')
def video():
    return render_template('video.html')

def gen():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    while(True):
        ret, frame = cap.read()
        if not ret:
            frame = cv2.VideoCapture(0)
            continue
        
        if ret:
            frame = np.asarray(frame)
            try:
                results = detector.detect_faces(frame)
                for i in range(len(results)):
                    x1, y1, width, height = results[i]['box']
                    x1, y1 = abs(x1), abs(y1)
                    frame = cv2.rectangle(frame,(x1,y1),(x1+width,y1+height),(0,255,0),2)
                                   
            except:
                print("Something else went wrong")
           
           
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break
    

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run( debug=True)