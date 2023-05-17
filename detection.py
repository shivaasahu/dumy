import numpy as np
import cv2

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

print(with_mask.shape)
print(without_mask.shape )
with_mask = with_mask.reshape(761,50*50*3)
without_mask = without_mask.reshape(1285,50*50*3)

print(with_mask.shape) ## = op 200,7500
print(without_mask.shape ) ## = op 200,7500

X = np.r_[with_mask , without_mask]
##X.shape ##op (400,7500)

lables = np.zeros(X.shape[0]) 
lables[761:] =1.0
names = {0:'Mask',1:'No Mask'}

##svm = support vector machine
##svc = support vector classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,lables,test_size=0.30)
print(x_train.shape)
## x_test = pca.transform(x_test)
## PCA = principal component analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_train = pca.fit_transform(x_train)

x_train,x_test,y_train,y_test = train_test_split(X,lables,test_size=0.25)


svm = SVC()
svm.fit(x_train,y_train)
##x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

print(accuracy_score(y_test , y_pred))

capture = cv2.VideoCapture(0)
harr_data = cv2.CascadeClassifier('data.xml')
font = cv2.FONT_HERSHEY_COMPLEX

while True:
   flag,img = capture.read()
   if flag : 
      faces = harr_data.detectMultiScale(img)
      for x,y,w,h in faces:
         cv2.rectangle(img,(x,y),(x+w , y+h),(255,0,0),4)
         face = img[y:y+h,x:x+w,:] 
         face = cv2.resize(face,(50,50))
         face = face.reshape(1,-1)
         ##face = pca.transform(face)
         pred = svm.predict(face)[0]
         n = names[int(pred)]
         cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
         print(n)
      cv2.imshow('window',img)
      if cv2.waitKey(2) == 27:
         break
capture.release()
cv2.destroyAllWindows()


