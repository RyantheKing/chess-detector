import cv2
import numpy as np
import time

class Square:
    def __init__(self, location, x_dict, y_dict, raw):
        self.single_dot = False

        self.piece = ' '
        self.location = location
        #print(location)
        self.coords = [ set(x_dict[location[1]]).intersection(set(y_dict[location[0]])).pop(),
                        set(x_dict[location[1]+1]).intersection(set(y_dict[location[0]])).pop(),
                        set(x_dict[location[1]+1]).intersection(set(y_dict[location[0]+1])).pop(),
                        set(x_dict[location[1]]).intersection(set(y_dict[location[0]+1])).pop()]
        print(self.coords)
        self.top_y = (self.coords[0][1] + self.coords[1][1])/2
        self.right_x = (self.coords[1][0] + self.coords[2][0])/2
        self.bot_y = (self.coords[2][1] + self.coords[3][1])/2
        self.left_x = (self.coords[3][0] + self.coords[0][0])/2
        self.rect_coords = [(self.left_x, self.top_y), (self.right_x, self.top_y), (self.left_x, self.bot_y), (self.right_x, self.bot_y)]
        print(self.bot_y, self.top_y, self.left_x, self.right_x)
        self.center = ((self.left_x+self.right_x)/2, (self.top_y+self.bot_y)/2)

        self.img = raw[int(self.bot_y):int(self.top_y), int(self.left_x):int(self.right_x)]
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        #self.gray_blur = cv2.GaussianBlur(self.gray, (41, 41), 0)
        self.thresh, self.bw_mask = cv2.threshold(self.gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #self.contours, self.hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #length = len(self.contours)
        if self.single_dot: self.color = [i for i in reversed(raw[int(self.center[1])][int(self.center[0])])]
        else:
            color_list=[]
            for i in range(int(self.center[1]-9), int(self.center[1]+10)):
                for i2 in range(int(self.center[0]-9), int(self.center[0]+10)):
                    color_list.append(raw[i][i2])
            self.color = [sum([i[2] for i in color_list])/len(color_list), sum([i[1] for i in color_list])/len(color_list), sum([i[0] for i in color_list])/len(color_list)]
        color_list=[]
        for i in range(int(self.bot_y)+10, int(self.top_y)-9):
            for i2 in range(int(self.left_x)+10, int(self.right_x)-10):
                color_list.append(raw[i][i2])
        self.whole_color = [sum([i[2] for i in color_list])/len(color_list), sum([i[1] for i in color_list])/len(color_list), sum([i[0] for i in color_list])/len(color_list)]
        self.intitial_whole_color = self.whole_color

    def update(self, raw):
        self.img = raw[int(self.bot_y):int(self.top_y), int(self.left_x):int(self.right_x)]
        if self.single_dot: self.color = [i for i in reversed(raw[int(self.center[1])][int(self.center[0])])]
        else:
            color_list=[]
            for i in range(int(self.center[1]-9), int(self.center[1]+10)):
                for i2 in range(int(self.center[0]-9), int(self.center[0]+10)):
                    color_list.append(raw[i][i2])
            self.color = [sum([i[2] for i in color_list])/len(color_list), sum([i[1] for i in color_list])/len(color_list), sum([i[0] for i in color_list])/len(color_list)]
        color_list=[]
        for i in range(int(self.bot_y)+10, int(self.top_y)-9):
            for i2 in range(int(self.left_x)+10, int(self.right_x)-10):
                color_list.append(raw[i][i2])
        self.whole_color = [sum([i[2] for i in color_list])/len(color_list), sum([i[1] for i in color_list])/len(color_list), sum([i[0] for i in color_list])/len(color_list)]
        
def notation(location):
    return (int(location[1]), ord(location[0].lower())-96)