import cv2

image = cv2.imread("image.png")

blurred = cv2.GaussianBlur(image, (3,3), 10)
blurred2 = cv2.GaussianBlur(image, (3,3), 1000)

cv2.imshow("Original Image", image)
cv2.imshow("blurred Image", blurred)
cv2.imshow("blurred2 Image", blurred2)
cv2.waitKey(0)
cv2.destroyAllWindows()