# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Fit a rotated rect
rotatedRect = cv2.minAreaRect(contours[0])
# Get rotated rect dimensions
(x, y), (width, height), angle = rotatedRect
# Get the 4 corners of the rotated rect
rotatedRectPts = cv2.boxPoints(rotatedRect)
rotatedRectPts = np.int0(rotatedRectPts)
# Draw the rotated rect on the image
out = color.copy()
cv2.drawContours(out, [rotatedRectPts], 0, (0, 255, 0), 2)
cv2.imwrite('output/minRect.png', out)