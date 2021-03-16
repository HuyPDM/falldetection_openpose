import cv2 as cv
#img = cv.imread('photo/1.png')
#cv.imshow('1',img) 
capture = cv.VideoCapture(1)
while True:
    isTrue, frame = capture.read()
    cv.imshow('video',frame)
    if cv.waitKey(20) & oxFF == ord('j'):
        break
capture.release()
cv.destroyAllWindows()
#cv.waitKey(0
