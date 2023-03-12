import cv2
import numpy as np

img = cv2.imread('test.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(img1,127,255,0)
contours,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# print("Number of contours detected:", len(contours))
cnt = contours[0]

# compute straight bounding rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.drawContours(img,[cnt],0,(255,255,0),2)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# compute rotated rectangle (minimum area)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

# draw minimum area rectangle (rotated rectangle)
img = cv2.drawContours(img,[box],0,(0,255,255),2)
cv2.imshow("Bounding Rectangles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
