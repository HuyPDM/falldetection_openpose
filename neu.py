
import time
import cv2 as cv
from flask import Flask, render_template, Response
import matplotlib.pyplot as plt 

config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
model = cv.dnn_DetectionModel(frozen_model,config)
classlabel =[]
file_label ='label.txt'
with  open(file_label,'rt') as f:
    classlabel = f.read().rstrip('\n').split('\n')
app = Flask(__name__)
sub = cv.createBackgroundSubtractorMOG2()  # create background subtractor



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    cap = cv.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
    model.setInputSize(320,320)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputScale(1.0/127.5)
    model.setInputSwapRB(True)
    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN
    # Read until video is completed
    while (cap.isOpened):
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        image = cv.resize(frame, (0, 0), None, 1, 1)  # resize image
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # converts image to gray
        fgmask = sub.apply(gray)  # uses the background subtraction
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        closing = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        dilation = cv.dilate(opening, kernel)
        classIndex, confident, box = model.detect(frame,confThreshold=0.55)
        print(classIndex)
        if (len(classIndex)!=0):
            for classID ,conF , boxes in zip(classIndex.flatten(), confident.flatten(), box) :
                if classIndex.all() == int(1):
                    cv.rectangle(image,boxes,(0,255,0))
                    cv.putText(image, 'person {}'.format(conF),(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)
        
        frame =cv.imencode('.jpg',image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cv.imshow('person detection',image)
        key = cv.waitKey(20)
        if key == 27:
           break
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=="__main__":
    app.run(host= "0.0.0.0")
    

