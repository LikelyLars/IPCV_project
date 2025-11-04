import cv2

image = cv2.imread("rainbow.png", cv2.IMREAD_UNCHANGED)
image = cv2.resize(image, (600, 400))

cv2.namedWindow("rainbow")
cv2.imshow("rainbow", image)
cv2.waitKey(0)
cv2.destroyAllWindows()