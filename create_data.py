import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy
import cv2

csv = pd.read_csv("data_path_female.txt",delim_whitespace="",header=None)
csv1 = pd.read_csv("data_path_male.txt",delim_whitespace="",header=None)

images_path,labels = csv[0],csv[1]

image_list=[]
label_list=[]

for i in range(len(labels)):
    image = cv2.imread(images_path[i],0)
    res = cv2.resize(image,(300,300))
    image_list.append(res)
    label_list.append(int(labels[i]))

images_path,labels = csv1[0],csv1[1]


for i in range(len(labels)):
    image = cv2.imread(images_path[i],0)
    res = cv2.resize(image,(300,300))
    image_list.append(res)
    label_list.append(int(labels[i]))

final_image_list = numpy.asarray(image_list)
final_label_list = numpy.asarray(label_list)


f = file("images.bin","wb")
g = file("labels.bin","wb")
numpy.save(f,final_image_list)
numpy.save(g,final_label_list)
