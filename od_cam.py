import cv2 as cv
import matplotlib.pyplot as plt 

config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

model = cv.dnn_DetectionModel(frozen_model,config)
classlabel =[]
file_label ='label.txt'
with  open(file_label,'rt') as f:
    classlabel = f.read().rstrip('\n').split('\n')
print(classlabel)

cap = cv.VideoCapture(-1)
cap.set(3,330)
cap.set(4,288)
model.setInputSize(320,320)
model.setInputMean((127.5,127.5,127.5))
model.setInputScale(1.0/127.5)
model.setInputSwapRB(True)
font_scale = 3
font = cv.FONT_HERSHEY_PLAIN
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    classIndex, confident, box = model.detect(frame,confThreshold=0.55)
    print(classIndex)
    if (len(classIndex)!=0):
        for classID ,conF , boxes in zip(classIndex.flatten(), confident.flatten(), box) :
            if classIndex.all() == int(1):
                cv.rectangle(frame,boxes,(0,255,0))
                cv.putText(frame, 'person {}'.format(conF),(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)
    cv.imshow('person detection',frame)
cap.release()
cv.destroyAllWindows()