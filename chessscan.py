import chess
import statistics
import cv2
import time
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import math
import funcs
import board

image = cv2.imread('chess2.png')

default_board =[['R','N','B','Q','K','B','N','R'],
                ['P','P','P','P','P','P','P','P'],
                [' ',' ',' ',' ',' ',' ',' ',' '],
                [' ',' ',' ',' ',' ',' ',' ',' '],
                [' ',' ',' ',' ',' ',' ',' ',' '],
                [' ',' ',' ',' ',' ',' ',' ',' '],
                ['p','p','p','p','p','p','p','p'],
                ['r','n','b','q','k','b','n','r']]

def set_position(board_dict, default_board):
    for index1 in range(1,9):
        for index2 in range(1,9):
            board_dict[str(index1)+str(index2)].piece = default_board[index1-1][index2-1]

def proper_board(old_white_count, old_black_count, board_list, whites_turn=True): #checks if the right number of pieces are on the board
    white_count = [x for l in board_list for x in l].count('W')
    black_count = [x for l in board_list for x in l].count('B')
    if whites_turn:
        if white_count==old_white_count and (black_count==old_black_count or black_count==old_black_count-1): return True, white_count, black_count
    else:
        if black_count==old_black_count and (white_count==old_white_count or white_count==old_white_count-1): return True, white_count, black_count
    return False, None, None

def piece_moved(prev_board_list, board_list, piece_list, whites_turn): #checks if a piece moved
    prev_arr = np.array(prev_board_list)
    arr = np.array(board_list)
    difference = prev_arr!=arr
    if np.count_nonzero(difference)==2:
        locations = np.where(difference)
        p1, p2 = (locations[0][0], locations[1][0]), (locations[0][1], locations[1][1])
        if whites_turn:
            if (arr[p1]=='W' and arr[p2]==' ' and prev_arr[p2]=='W' and (prev_arr[p1]==' ' or prev_arr[p1]=='B')): #from square=p2; to square=p1
                piece_list[p1[0]][p1[1]] = piece_list[p2[0]][p2[1]]
                piece_list[p2[0]][p2[1]] = ' '
                return True, p2, p1
            elif (arr[p2]=='W' and arr[p1]==' ' and prev_arr[p1]=='W' and (prev_arr[p2]==' ' or prev_arr[p2]=='B')):  #from square=p1; to square=p2
                piece_list[p2[0]][p2[1]] = piece_list[p1[0]][p1[1]]
                piece_list[p1[0]][p1[1]] = ' '
                return True, p1, p2
        else:
            if (arr[p1]=='B' and arr[p2]==' ' and prev_arr[p2]=='B' and (prev_arr[p1]==' ' or prev_arr[p1]=='W')):  #from square=p2; to square=p1
                piece_list[p1[0]][p1[1]] = piece_list[p2[0]][p2[1]]
                piece_list[p2[0]][p2[1]] = ' '
                return True, p2, p1
            elif (arr[p2]=='B' and arr[p1]==' ' and prev_arr[p1]=='B' and (prev_arr[p2]==' ' or prev_arr[p2]=='W')): #from square=p1; to square=p2
                piece_list[p2[0]][p2[1]] = piece_list[p1[0]][p1[1]]
                piece_list[p1[0]][p1[1]] = ' '
                return True, p1, p2
    return False, None, None

def valid_move(chess_board, from_square, to_square):
    from_num = from_square[0]*8+from_square[1]
    to_num = to_square[0]*8+to_square[1]
    return chess.Move(from_num, to_num) in chess_board.pseudo_legal_moves, from_num, to_num


def initialize_board(x_vals, y_vals, raw):
    board_dict = {}
    for index1 in range(1,9):
        for index2 in range(1,9):
            board_dict[str(index1)+str(index2)] = board.Square((index1, index2), x_vals, y_vals, raw)
    return board_dict

def list_board(board_dict):
    board_format_list = [[],[],[],[],[],[],[],[]]
    board_pieces_list = [[],[],[],[],[],[],[],[]]
    for index1 in range(1,9):
        for index2 in range(1,9):
            board_pieces_list[index1-1].append(board_dict[str(index1)+str(index2)].piece)
            color_bool_list = [abs(board_dict[str(index1)+str(index2)].intitial_whole_color[i]-board_dict[str(index1)+str(index2)].color[i])<5 for i in range(3)]
            if color_bool_list.count(True) >= len(color_bool_list)*4/5:
                board_format_list[8-index1].append(' ')
            else:
                if sum(board_dict[str(index1)+str(index2)].color)>400: board_format_list[8-index1].append('W')
                else: board_format_list[8-index1].append('B')
    return board_format_list, board_pieces_list

def print_board(board_format_list):
    for i in board_format_list:
        for val in i:
            print(val, end=' ')
        print('')

def board_update(board_dict, img):
    for index1 in range(1,9):
        for index2 in range(1,9):
            board_dict[str(index1)+str(index2)].update(img)

def initial_frame_analysis(img):
    new_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))

    # Canny Edge detection
    sigma=100000
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower,upper)
    edges = cv2.Canny(img, 0, 900)
    cv2.imshow('win0', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Hough line detection
    min_line_length=0
    max_line_gap=3.14
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120, min_line_length, max_line_gap)

    lines = np.reshape(lines, (-1, 2))

    #print(lines)
    
    for rho,theta in lines:
        #print(float(rho), float(theta))
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        #cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imshow('win10', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # seperate to horizontal and vertical lines
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])

    # Locate intersections
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    intersection_points = np.array(points)

    # Cluster points
    dists = spatial.distance.pdist(intersection_points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    points = sorted(list(clusters), key=lambda k: [k[1], k[0]])

    dimensions = img.shape

    new_points = []

    for point in points:
        if dimensions[1]*0.02 < point[0] < dimensions[1]-dimensions[1]*0.02 and dimensions[0]*0.02 < point[1] < dimensions[0]-dimensions[0]*0.02:
            new_points.append(point)
            img = cv2.circle(img, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)

    center = (statistics.mean([i[0] for i in new_points]), statistics.mean([i[1] for i in new_points]))

    top_left_max = 0
    top_left_coords = (0,0)
    bottom_left_max = 0
    bottom_left_coords = (0,0)
    top_right_max = 0
    top_right_coords = (0,0)
    bottom_right_max = 0
    bottom_right_coords = (0,0)
    for i in new_points:
        if i[0]<center[0] and i[1]<center[1]:
            hypot = math.hypot(i[0]-center[0], i[1]-center[1])
            if hypot > top_left_max:
                top_left_max = hypot
                top_left_coords = i
        elif i[0]<center[0] and i[1]>center[1]:
            hypot = math.hypot(i[0]-center[0], i[1]-center[1])
            if hypot > bottom_left_max:
                bottom_left_max = hypot
                bottom_left_coords = i
        elif i[0]>center[0] and i[1]<center[1]:
            hypot = math.hypot(i[0]-center[0], i[1]-center[1])
            if hypot > top_right_max:
                top_right_max = hypot
                top_right_coords = i
        else:
            hypot = math.hypot(i[0]-center[0], i[1]-center[1])
            if hypot > bottom_right_max:
                bottom_right_max = hypot
                bottom_right_coords = i

    img = cv2.circle(img, (int(top_left_coords[0]), int(top_left_coords[1])), radius=3, color=(255, 0, 0), thickness=-1)
    img = cv2.circle(img, (int(bottom_left_coords[0]), int(bottom_left_coords[1])), radius=3, color=(255, 0, 0), thickness=-1)
    img = cv2.circle(img, (int(top_right_coords[0]), int(top_right_coords[1])), radius=3, color=(255, 0, 0), thickness=-1)
    img = cv2.circle(img, (int(bottom_right_coords[0]), int(bottom_right_coords[1])), radius=3, color=(255, 0, 0), thickness=-1)
    img = cv2.circle(img, (int(statistics.mean([i[0] for i in new_points])), int(statistics.mean([i[1] for i in new_points]))), radius=3, color=(0, 255, 0), thickness=-1)

    pts_src = np.array([[top_left_coords[0], top_left_coords[1]],[top_right_coords[0], top_right_coords[1]],[bottom_right_coords[0],bottom_right_coords[1]],[bottom_left_coords[0], bottom_left_coords[1]]])
    #---- 4 corner points of the black image you want to impose it on
    pts_dst = np.array([[0.0,0.0],[800.0, 0.0],[800.0,800.0],[0.0, 800.0]])
    #---- forming the black image of specific size
    im_dst = np.zeros((800, 800, 3), np.uint8)
    #---- Framing the homography matrix
    h, status = cv2.findHomography(pts_src, pts_dst)
    #---- transforming the image bound in the rectangle to straighten
    im_out = cv2.warpPerspective(img, h, (im_dst.shape[1],im_dst.shape[0]))
    raw_out = cv2.warpPerspective(new_img, h, (im_dst.shape[1],im_dst.shape[0]))
    #pts_src = np.array([[17.0,0.0], [77.0,5.0], [0.0, 552.0],[53.0, 552.0]])

    temp = cv2.perspectiveTransform(np.array([new_points], dtype=np.float32), h, (im_dst.shape[1],im_dst.shape[0]))

    warped_points = temp.tolist()[0]

    for point in warped_points[:4]:
        im_out = cv2.circle(im_out, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)

    new_dimensions = im_out.shape

    x_dict = {}
    y_dict = {}

    for i in warped_points:
        found = False
        for val in x_dict:
            if val-new_dimensions[1]*0.05 < i[0] < val+new_dimensions[1]*0.05:
                found = True
                x_dict[val].append(tuple(i))
        if not found:
            x_dict[i[0]] = [tuple(i)]

        found = False
        for val in y_dict:
            if val-new_dimensions[0]*0.05 < i[1] < val+new_dimensions[0]*0.05:
                found = True
                y_dict[val].append(tuple(i))
        if not found:
            y_dict[i[1]] = [tuple(i)]

    x_lines = []
    for i in x_dict:
        line = np.polyfit([val[0] for val in x_dict[i]], [val[1] for val in x_dict[i]], 1)
        x_lines.append(line)
        cv2.line(im_out,(int(-line[1]/line[0]),0),(int((dimensions[0]-line[1])/line[0]),dimensions[0]),(0,0,255),2)

    y_lines = []
    for i in y_dict:
        line = np.polyfit([val[0] for val in y_dict[i]], [val[1] for val in y_dict[i]], 1)
        y_lines.append(line)
        cv2.line(im_out,(0,int(line[1])),(dimensions[1],int(line[0]*dimensions[1]+line[1])),(0,0,255),2)
    #y=mx+b or x=(y-b)/m
    #print(x_dict)

    x_keys = list(x_dict.keys())
    y_keys = list(y_dict.keys())
    x_keys.sort()
    y_keys.sort()

    #print(x_keys,y_keys)
    if len(x_dict) == 11 and len(y_dict) == 11:
        x_keys.pop(10)
        x_keys.pop(0)
        y_keys.pop(10)
        y_keys.pop(0)

    x_vals={}
    count = 1#len(x_keys)
    for i in x_keys:
        x_vals[count] = x_dict[i]
        count+=1

    count = len(y_keys)
    y_vals={}
    for i in y_keys:
        y_vals[count] = y_dict[i]
        count-=1

    raw_gray = cv2.cvtColor(raw_out, cv2.COLOR_BGR2GRAY)
    raw_gray_blur = cv2.blur(raw_gray, (5, 5))

    thresh, bw_mask = cv2.threshold(raw_gray_blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #(y-b)/m= x

    return (im_out, raw_out, img), x_vals, y_vals, h, (im_dst.shape[1], im_dst.shape[0])

#initialize video
video = cv2.VideoCapture(0)
ret, frame = video.read()
frame = cv2.imread('chess2.png')

# intialize board
images, x_vals, y_vals, h, dsize  = initial_frame_analysis(frame)
cv2.imshow('win1', images[0])
cv2.imshow('win2', images[1])
cv2.imshow('win3', images[2])
cv2.waitKey(0)
cv2.destroyAllWindows()
board_dict = initialize_board(x_vals, y_vals, images[1])
board_list, board_pieces_list = list_board(board_dict)
print_board(board_list)
empty_board = board_dict.copy()
set_position(board_dict, default_board)

# wait for setup
found_starting = False
while not found_starting:
    ret, frame = video.read()
    frame = cv2.imread('chess2.png')
    raw_out = cv2.warpPerspective(frame, h, dsize)
    board_update(board_dict, raw_out)
    board_list, board_pieces_list = list_board(board_dict)
    print(board_list)
    if board_list==[['W','W','W','W','W','W','W','W'],
                    ['W','W','W','W','W','W','W','W'],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    ['B','B','B','B','B','B','B','B'],
                    ['B','B','B','B','B','B','B','B']]:
        found_starting = True
initial_board = board_dict.copy()
print_board(board_list)
chess_board = chess.Board(chess.STARTING_FEN)

white_count = 16
black_count = 16
last_valid_board_list = board_list
whites_turn = True
while True:
    ret, frame = video.read()
    frame = image
    raw_out = cv2.warpPerspective(frame, h, dsize)
    board_update(board_dict, raw_out)
    board_list, board_pieces_list = list_board(board_dict)
    # Check if the board position is valid giving previous pieces
    status_result = proper_board(white_count, black_count, board_list, whites_turn)
    if status_result[0]:
        # Check if a proper piece actually moved location
        moved, from_square, to_square = piece_moved(last_valid_board_list, board_list, board_pieces_list, whites_turn)
        if moved: 
            # Check if the change is a valid chess move
            valid, from_num, to_num = valid_move(chess_board, from_square, to_square)
            if valid:
                chess_board.push(chess.Move(from_num, to_num))
                set_position(board_dict, board_pieces_list)
                last_valid_board_list = board_list.copy()
                white_count, black_count = status_result[1:]
                whites_turn = not whites_turn

#print('Completed!')
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------


"""
img = cv2.imread('checkers.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.blur(gray, (5, 5))

# Canny Edge detection
sigma=0.33
v = np.median(img)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(img, lower, upper)

# Hough line detection
min_line_length=100
max_line_gap=10
lines = cv2.HoughLines(edges, 1, np.pi / 180, 80, min_line_length, max_line_gap)

lines = np.reshape(lines, (-1, 2))

for rho,theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# seperate to horizontal and vertical lines
h_lines, v_lines = [], []
for rho, theta in lines:
    if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
        v_lines.append([rho, theta])
    else:
        h_lines.append([rho, theta])

# Locate intersections
points = []
for r_h, t_h in h_lines:
    for r_v, t_v in v_lines:
        a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
        b = np.array([r_h, r_v])
        inter_point = np.linalg.solve(a, b)
        points.append(inter_point)
intersection_points = np.array(points)

for point in points:
    img = cv2.circle(img, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-1)

# Cluster points
dists = spatial.distance.pdist(points)
single_linkage = cluster.hierarchy.single(dists)
flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
cluster_dict = defaultdict(list)
for i in range(len(flat_clusters)):
    cluster_dict[flat_clusters[i]].append(points[i])
cluster_values = cluster_dict.values()
clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
points = sorted(list(clusters), key=lambda k: [k[1], k[0]])

dimensions = img.shape

new_points = []

for point in points:
    if dimensions[1]*0.05 < point[0] < dimensions[1]-dimensions[1]*0.05 and dimensions[0]*0.05 < point[1] < dimensions[0]-dimensions[0]*0.05:
        new_points.append(point)
        img = cv2.circle(img, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-1)

cv2.imshow('raw', img)
cv2.imshow('gray', edges)
cv2.imshow('gray_blur', gray_blur)

print('Completed!')
cv2.waitKey(0)
cv2.destroyAllWindows()
"""