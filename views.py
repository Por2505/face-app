from flask import render_template, request
from flask import redirect,url_for
from PIL import Image
import os
import cv2
from exface import extract_face
from faces import facenett
import os

UPLOAD_FLODER = 'static/uploads'
def base():
    return render_template("base.html")
def login():
    return render_template("login.html")

def index():
    return render_template("index.html")


def faceapp():
    return render_template("faceapp.html")


def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def getname():
    if request.method == 'POST':
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        name=facenet(path,filename)     
    return name
    
    

def gender():
    if request.method == 'POST':
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        px = extract_face(path)
        cv2.imwrite('./static/predict/{}'.format(filename),px)
        #pipeline_model(path,filename,color='bgr')
        name = facenett(path,filename)
        folder_path = (r'C:\Users\Por\Desktop\faces\Face App\data')
        test = os.listdir(folder_path)
        for images in test:
            if images.endswith(".jpg"):
                os.remove(os.path.join(folder_path, images))

        return render_template('gender.html',fileupload=True,img_name=filename, w=w,name=name)
    return render_template('gender.html',fileupload=False,img_name="freeai.png")