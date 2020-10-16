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
    # rendering webpage
    return render_template('video.html')
# def gen(camera):
#     while True:
#         #get camera frame
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(video),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen(video):
#     while True:
#         success, image = video.read()
#         ret, jpeg = cv2.imencode('.jpg', image)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def gen():
    cap = cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    ds_factor=1.0
    while(cap.isOpened()):
        ret, frame = cap.read()
        #detector = MTCNN()
        # if not ret:
        #     frame = cv2.VideoCapture(0)
        #     continue
        
        # if ret:
            
        #     boxes = detector.detect_faces(frame)
        #     frame_draw = frame.copy()
        #     image = Image.fromarray(frame_draw)
        #     draw = ImageDraw.Draw(image)


        #     for box in boxes:
        #         draw.rectangle(box["box"], outline=(255, 0, 0), width=4)
        #     image = cv2.cvtColor(np.array(image)[..., ::-1], cv2.COLOR_RGB2BGR)
        #     image = cv2.resize(image, (640, 480))

        if ret:
            image=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)                    
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            face_rects=face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in face_rects:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                break

        frame = cv2.imencode('.jpg', image)[1].tobytes()
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