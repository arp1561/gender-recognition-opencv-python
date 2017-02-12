import cv2
import numpy as np

image_file = file("images.bin","rb")
labels_file = file("labels.bin","rb")

images = np.load(image_file)
labels = np.load(labels_file)


model = cv2.createFisherFaceRecognizer()
model.train(images,labels)


'''
sample = cv2.imread("sample5.jpg",0)
sample = cv2.resize(sample,(300,300))
answer = model.predict(sample)
print answer
'''

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(300,300))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    answer = model.predict(gray)
    if answer[0]==0:
        print "Female"
    else:
        print "Male"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

