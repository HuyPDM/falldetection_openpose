import cv2 as cv
import matplotlib.pyplot as plt 
from imutils.video import FPS
import numpy as np
import os
useGPU=False
config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
net = cv.dnn.readNetFromTensorflow(frozen_model,config)

classlabel =[]
file_label ='label.txt'
with  open(file_label,'rt') as f:
    classlabel = f.read().rstrip('\n').split('\n')
#print(classlabel)

if (useGPU):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH , 300);
cap.set(cv.CAP_PROP_FRAME_HEIGHT , 300);
writer =None
fps = FPS().start()
while True:
    (grap, frame )= cap.read()
    if not grap:
        break
    blob = cv.dnn.blobFromImage(frame,size= (300, 300),swapRB= True,crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final","detection_masks"])
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
        if confidence > 0.55:
			# scale the bounding box coordinates back relative to the
			# size of the frame and then compute the width and the
			# height of the bounding box
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
            mask = masks[i, classID]
            mask = cv.resize(mask, (boxW, boxH),
                interpolation=cv.INTER_CUBIC)
            mask = (mask > 0.55)
            # extract the ROI of the image but *only* extracted the
			# masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]
			# grab the color used to visualize this particular class,
			# then create a transparent overlay by blending the color
			# with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
			# store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended
			# draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv.rectangle(frame, (startX, startY), (endX, endY),color, 2)
			# draw the predicted label and associated probability of
			# the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv.putText(frame, text, (startX, startY - 5),cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # screen
		# show the output frame
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
    # if args["output"] != "" and writer is None:
		# initialize our video writer
        # fourcc = cv.VideoWriter_fourcc(*"MJPG")
        # writer = cv.VideoWriter(args["output"], fourcc, 30,
        # (frame.shape[1], frame.shape[0]), True)
	# if the video writer is not None, write the frame to the output
	# video file
    if writer is not None:
        writer.write(frame)
	# update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))