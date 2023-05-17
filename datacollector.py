import os
import cv2
import numpy as np

harr_data = cv2.CascadeClassifier('data.xml')

path_with_mask = 'dataset/with_mask'
path_without_mask = 'dataset/without_mask'

images_with_mask = os.listdir(path_with_mask)
images_without_mask = os.listdir(path_without_mask)

data_with_mask = []
for img in images_with_mask:
    with_mask = cv2.imread(os.path.join(path_with_mask, img))
    faces = harr_data.detectMultiScale(with_mask)
    for (x, y, w, h) in faces:
        face = with_mask[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50))
        data_with_mask.append(face)
np.save('with_mask.npy', data_with_mask)

data_without_mask = []
for img in images_without_mask:
    without_mask = cv2.imread(os.path.join(path_without_mask, img))
    faces = harr_data.detectMultiScale(without_mask)
    for (x, y, w, h) in faces:
        face = without_mask[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50))
        data_without_mask.append(face)
np.save('without_mask.npy', data_without_mask)

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')
