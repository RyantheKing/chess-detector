# List the output points in the same order as input
# Top-left, top-right, bottom-right, bottom-left
dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
# Get the transform
m = cv2.getPerspectiveTransform(np.float32(intersect_pts), np.float32(dstPts))
# Transform the image
out = cv2.warpPerspective(color, m, (int(width), int(height)))
# Save the output
cv2.imwrite('output/page.png', out)