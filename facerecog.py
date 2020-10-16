#from keras.models import load_model
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image


data = np.load('5-celebrity-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# load the facenet model
facenet_model = load_model('./facenet_keras.h5')
#print('Loaded Model')


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    
    return face_array

def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]
    
def load_face(dir):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


emdTrainX = list()
for face in trainX:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)
    
emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

def facen(path,filename):
    px = extract_face(path)
    g = cv2.cvtColor(px,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./static/predict/{}'.format(filename),g)
    em = get_embedding(facenet_model, px)

    emdTest=list()
    emdTest.append(em)
    emdTest = np.asarray(emdTest)
    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)
    emdTest_norm = in_encoder.transform(emdTest)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy_enc = out_encoder.transform(trainy)
    model = SVC(kernel='linear', probability=True)
    model.fit(emdTrainX_norm, trainy_enc)
# predict
    yhat_train = model.predict(emdTrainX_norm)
    yhat_test = model.predict(emdTest_norm)
    score_train = accuracy_score(trainy_enc, yhat_train)
    face = px
    face_emd = emdTest_norm[0]

# prediction for the face
    sample = np.expand_dims(face_emd, axis=0)
    yhat_c = model.predict(sample)
    yhat_p = model.predict_proba(sample)
# get name
    class_in = yhat_c[0]
    class_prob = yhat_p[0,class_in] * 100
    predict_name = out_encoder.inverse_transform(yhat_c)
    all_names = out_encoder.inverse_transform([0,1,2,3,4])
    print('Predicted: %s (%.3f)' % (predict_name[0], class_prob))
    print('Predicted: \n%s \n%s' % (all_names, yhat_p[0]*100))
    print('Expected: %s' % predict_name)
    #cv2.imwrite('./static/predict/1.jpg',random_face)
    return predict_name[0]