import cv2
import numpy as np

image_file = file("images.bin","rb")
labels_file = file("labels.bin","rb")

images = np.load(image_file)
labels = np.load(labels_file)


model = cv2.createFisherFaceRecognizer()
model.train(images,labels)


sample = cv2.imread("testData/sample.jpg",0)
sample = cv2.resize(sample,(300,300))
answer = model.predict(sample)
print answer

'''
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(300,300))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    answer = model.predict(gray)
    if answer[0]==0:
        print "Female"
        cv2.putText(frame,'Female',(10,500), font, 4,(255,255,255),2,cv2.CV_AA)
    else:
        print "Male"
        cv2.putText(frame,'Male',(10,500), font, 4,(255,255,255),2,cv2.CV_AA)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
