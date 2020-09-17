from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import cv2

def draw_image_with_boxes(path, result_list):
    img = cv2.imread(path)
    #cv2.imshow(data)
    #ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        #add_patch(rect)
    for key, value in result['keypoints'].items():
        dot = Circle(value, radius=2, color='red')
        #add_patch(dot)
    
#pyplot.show()
#filename = 'f.jpg'
#pixels = pyplot.imread(filename)
#detector = MTCNN()
#faces = detector.detect_faces(pixels)
#draw_image_with_boxes(filename, faces)
