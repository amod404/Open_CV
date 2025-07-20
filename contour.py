import cv2
import numpy as np

img = cv2.imread("image.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)

    # print(approx)
    corners = len(approx)
    shape_name="none"

    if corners == 3:
        shape_name = "Triangle"
    elif corners == 4:
        shape_name = "Quadrilateral"
    elif corners >5:
        shape_name = "Circle"
    
    cv2.drawContours(img, [approx], 0,(0,255,0),2)
    x = int(np.mean(approx.ravel()[::2]))
    y = int(np.mean(approx.ravel()[1::2]))
    cv2.putText(img, shape_name, (x-30,y),cv2.FONT_HERSHEY_COMPLEX, 0.6 ,(255,255,255))



cv2.imshow("title",img)
cv2.waitKey(0)
cv2.destroyAllWindows()