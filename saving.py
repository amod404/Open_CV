import cv2

image = cv2.imread("image.png")

(h,w) = image.shape[:2]

center = (w//2,h//2)

M = cv2.getRotationMatrix2D((h//2,h//2),90,1.0)
rotated = cv2.warpAffine(image,M,(h,w))

cv2.imshow("title", image)
cv2.imshow("title2", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()