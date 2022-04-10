import cv2

img = cv2.imread('im_out.png')


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, corners = cv2.findChessboardCorners(img, (9,9))

corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)

cv2.drawChessboardCorners(img, (7,6), corners2, ret)

cv2.imshow('raw', img)

print('Completed!')
cv2.waitKey(0)
cv2.destroyAllWindows()