# Detect lines using hough transform
polar_lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
drawHoughLines(color, polar_lines, 'output/houghlines.png')
# Detect the intersection points
intersect_pts = lq.hough_lines_intersection(polar_lines, gray.shape)
# Sort the points in cyclic order
intersect_pts = cyclic_intersection_pts(intersect_pts)
# Draw intersection points and save
out = color.copy()
for pts in intersect_pts:
    cv2.rectangle(out, (pts[0] - 1, pts[1] - 1), (pts[0] + 1, pts[1] + 1), (0, 0, 255), 2)
cv2.imwrite('output/intersect_points.png', out)