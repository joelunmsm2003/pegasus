import cv2
import numpy as np
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')



cap =cv2.VideoCapture(0)

while True:

	ret,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:

		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=img[y:y+h,x:x+w]
		eyes=eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow('img',img)


	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


#Capitulo 15
# cap =cv2.VideoCapture(0)
# fgbg=cv2.createBackgroundSubtractorMOG2()

# while True:

# 	ret,frame=cap.read()
# 	fgmask=fgbg.apply(frame)

# 	cv2.imshow('original',frame)
# 	cv2.imshow('fg',fgmask)

# 	k=cv2.waitKey(30) & 0xFF
# 	if k==27:

# 		break
# cap.release()
# cv2.destroyAllWindows()
#ccc
# img_bgr=cv2.imread('fresas.jpg')
# img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

# template =cv2.imread('fresa.jpg',0)
# w,h=template.shape[::-1]

# res=cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold=0.2
# loc=np.where(res>=threshold)

# for pt in zip(*loc[::-1]):

# 	print('entre....')
# 	cv2.rectangle(img_bgr,pt,(pt[0]+w,pt[1]+h),(0,255,255),2)


# cv2.imshow('detected',img_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Cap 10
# cap =cv2.VideoCapture(0)

# while True:

# 	_,frame = cap.read()
# 	laplacian =cv2.Laplacian(frame,cv2.CV_64F)
# 	sobelx=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
# 	sobely=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
# 	edges=cv2.Canny(frame,100,150)

# 	cv2.imshow('original',frame)
# 	cv2.imshow('laplacian',laplacian)
# 	#cv2.imshow('sobelx',sobelx)
# 	#cv2.imshow('sobely',sobely)
# 	cv2.imshow('edges',edges)

# 	if cv2.waitKey(1) & 0xFF==ord('q'):
		
# 		break


# cv2.destroyAllWindows()
# cap.release()


#Cap8
# 	_,frame=cap.read()
# 	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

# 	lower_red=np.array([150,150,50])
# 	upper_red=np.array([255,255,255])

# 	mask=cv2.inRange(hsv,lower_red,upper_red)
# 	res=cv2.bitwise_and(frame,frame,mask=mask)

# 	cv2.imshow('frame',frame)
# 	cv2.imshow('mask',mask)
# 	cv2.imshow('res',res)
	


# 	if cv2.waitKey(1) & 0xFF==ord('q'):
# 		break


# cv2.destroyAllWindows()
# cap.release()
#Cap5 
# img1 =cv2.imread('girl.jpg')
# img2 =cv2.imread('dog.jpeg')

# #add=cv2.add(img1,img2)

# weighted=cv2.addWeighted(img1,0,img2,0.4,0)


# cv2.imshow('weighted',weighted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#Cap 4
# img =cv2.imread('girl.jpg',cv2.IMREAD_COLOR)

# img[55,55]=[255,255,255]
# px=img[55,55]
# print(px)

# img[100:150, 100:150]=[255,255,255]

# watch_face=img[10:23,10:19]
# img[30:74, 30:87]=watch_face


# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(rot)

#Cap 3
# img =cv2.imread('girl.jpg',cv2.IMREAD_COLOR)

# cv2.line(img,(0,0),(150,150),(255,255,255),15)

# cv2.rectangle(img,(15,25),(200,150),(0,255,0),5)

# cv2.circle(img,(100,63),55,(0,0,255),-1)


# pts =np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
# cv2.polylines(img,[pts],True,(0,255,255),3)


# font =cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'kdkdkdk',(0,130),font,1,(200,22,22),5,cv2.LINE_AA)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Cap 2

# cap =cv2.VideoCapture(0)
# fourcc =cv2.VideoWriter_fourcc(*'XVID')
# out =cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))


# while True:
# 	ret,frame=cap.read()
# 	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 	out.write(frame)
# 	cv2.imshow('frame',frame)
# 	cv2.imshow('gray',gray)
# 	if cv2.waitKey(1) & 0xFF==ord('q'):
# 		break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

#Cap1


#img =cv2.imread('girl.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#plt.imshow(img,cmap='gray',interpolation='bicubic')
#plt.plot([50, 100],[800,100],'c',linewidth=5)
#plt.show()