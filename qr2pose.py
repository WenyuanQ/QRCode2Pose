import cv2
import numpy as np
import pyzbar
from pyzbar.pyzbar import decode
import random
import time

def get_distance(point1,point2):
    return ((point1[0] - point2[0])**2 + (point2[1] -point2[1])**2)**0.5

def get_center(poly):

    return (cX,cY)

def n2t(array):
    return tuple(array.astype(np.int).tolist())

_color_index = 0
_colors = [(255,50,50),(50,255,50),(50,50,255),(255,255,50),(50,255,255),(255,50,255)]
random.shuffle(_colors)
def random_color():
    global _color_index
    _color_index += 1
    if _color_index == len(_colors):
        _color_index = 0
    return _colors[_color_index]

class QR_Finder(object):
    def __init__(self,poly_l,poly_m,poly_s):
        self.poly_l = poly_l
        self.poly_m = poly_m
        self.poly_s = poly_s
        self._calc_center()

    def _calc_center(self):
        x,y,counter = 0,0,0
        for poly in [self.poly_l,self.poly_m,self.poly_s]:
            for point in poly:
                x += point[0][0]
                y += point[0][1]
                counter += 1
        self.center = np.array([x/counter,y/counter],dtype=np.int)

    def calc_corner(self,qr_center):
        vec = []
        for point in self.poly_l:
            vec.append(point[0]-self.center)
        qr_finder_vec = qr_center - self.center
        cross = []
        for v in vec:
            cross.append(abs(np.cross(v,qr_finder_vec)))
        sorted_index = np.argsort(cross)
        id1 = sorted_index[0]
        id2 = sorted_index[1]
        distance1 = np.linalg.norm(self.poly_l[id1][0]-qr_center)
        distance2 = np.linalg.norm(self.poly_l[id2][0]-qr_center)

        if distance1 > distance2:
            self.corner = self.poly_l[id1][0]
        else:
            self.corner = self.poly_l[id2][0]
        return self.corner

    def draw(self,image,color = (0,255,0), thickness = 1):
        cv2.drawContours(image, [self.poly_l,self.poly_m,self.poly_s], -1, color, thickness)
        cv2.circle(image,tuple(self.center),3,color,-1)

class QR_Code(object):
    def __init__(self,contour):
        self.contour = contour
        self.finders = []
        self.valid = False

    def calc(self):
        #self.polyDP = cv2.approxPolyDP(self.contour,0.03*cv2.arcLength(self.contour,True),True)
        M = cv2.moments(self.contour)
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        self.center = np.array([cX,cY],dtype=np.int)
        vec1 = self.finders[0].center - self.center
        vec2 = self.finders[1].center - self.center
        vec3 = self.finders[2].center - self.center
        cross1 = abs(np.cross(vec1,vec2))
        cross2 = abs(np.cross(vec1,vec3))
        cross3 = abs(np.cross(vec2,vec3))
        if cross1 < cross2 and cross1 < cross3:
            top_left_id = 2
            side1,side2 = [0,1]
        elif cross2 < cross1 and cross2 < cross3:
            top_left_id = 1
            side1,side2 = [0,2]
        else:
            top_left_id = 0
            side1,side2 = [1,2]

        self.finder_base = self.finders[top_left_id]
        self.finder_side1 = self.finders[side1]
        self.finder_side2 = self.finders[side2]
        self.center = (self.finder_side1.center + self.finder_side2.center)/2
        self.finder_oppo = self.center *2 - self.finders[top_left_id].center

        self.top_left = self.finder_base.calc_corner(self.center)
        self.top_right = self.finder_side1.calc_corner(self.center)
        self.bottom_left = self.finder_side2.calc_corner(self.center)

        bottom_right1 = self.center * 2 - self.top_left
        bottom_right2 = (self.top_right - self.top_left) + (self.bottom_left - self.top_left) + self.top_left
        self.bottom_right_guess = (bottom_right1+bottom_right2)/2

        distances = []
        for pt in self.contour:
            distances.append(np.linalg.norm(np.array(pt[0])-self.bottom_right_guess))
        sort_index = np.argsort(np.array(distances))
        self.bottom_right = self.contour[sort_index[0]][0]
        
        bonding_box = [[self.top_left],[self.top_right],[self.bottom_right],[self.bottom_left]]
        self.bonding_box = np.array(bonding_box,dtype=np.int)

        self.top_left2 = 1.1*(self.top_left-self.center) + self.center
        self.top_right2 = 1.1*(self.top_right-self.center) + self.center
        self.bottom_left2 = 1.1*(self.bottom_left-self.center) + self.center
        self.bottom_right2 = 1.1*(self.bottom_right-self.center) + self.center
        bonding_box2 = [[self.top_left2],[self.top_right2],[self.bottom_right2],[self.bottom_left2]]
        self.bonding_box2 = np.array(bonding_box2,dtype=np.int)

    def add_finder(self,finder):
        if cv2.pointPolygonTest(self.contour,tuple(finder.center),False) >= 0:
            self.finders.append(finder)
            if len(self.finders) == 3:
                self.valid = True
                self.calc()
            if len(self.finders) >= 4:
                self.valid = False
                print('[Waring] Too many finders for QR code')

    def draw(self,image,color, thickness = 1):
        for poly in self.finders:
            poly.draw(image,color = color, thickness = 1)
        if self.valid:
            cv2.drawContours(image,[self.bonding_box],0,color,2)
            cv2.drawContours(image,[self.bonding_box2],0,color,2)

            cv2.circle(image,n2t(self.center),5,(255,255,255),-1)

            cv2.circle(image,n2t(self.finder_base.center),5,(255,255,255),2)
            cv2.circle(image,n2t(self.finder_side1.center),5,(255,255,255),2)
            cv2.circle(image,n2t(self.finder_side2.center),5,(255,255,255),2)
            cv2.circle(image,n2t(self.finder_oppo),5,(255,255,255),2)
            cv2.line(image,n2t(self.finder_base.center),n2t(self.finder_side1.center),color,1)
            cv2.line(image,n2t(self.finder_base.center),n2t(self.finder_side2.center),color,1)
            cv2.line(image,n2t(self.finder_base.center),n2t(self.finder_oppo),color,1)
            cv2.line(image,n2t(self.finder_side1.center),n2t(self.finder_side2.center),color,1)
            cv2.line(image,n2t(self.finder_oppo),n2t(self.finder_side1.center),color,1)
            cv2.line(image,n2t(self.finder_oppo),n2t(self.finder_side2.center),color,1)

            cv2.circle(image,n2t(self.top_left),5,(255,255,0),-1)
            cv2.circle(image,n2t(self.bottom_right),5,(0,255,255),-1)
            cv2.circle(image,n2t(self.top_right),5,(0,255,0),-1)
            cv2.circle(image,n2t(self.bottom_left),5,(0,255,0),-1)
            #cv2.line(image,tuple(self.top_left.tolist()),tuple(self.bottom_right.tolist()),(0,255,0),2)

    def detect_qr(self,binary,index):
        mask = np.zeros(shape=binary.shape,dtype=np.uint8)
        cv2.fillPoly(mask, [self.contour], (255))
        pts1 = np.float32([self.top_left2,self.top_right2,self.bottom_left2,self.bottom_right2])
        size = 150
        pts2 = np.float32([[0,0],[size,0],[0,size],[size,size]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(binary,M,(size,size))

        qr = decode(dst)
        if [] == qr:
            cv2.imshow('dst_%d'%(index),dst)
            print('fail')
        else:
            print(qr[0].data)



def qr_code_detect(src):
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img_width,img_height = gray.shape

    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    process1 = np.zeros(shape=src.shape,dtype=src.dtype)
    edges = cv2.Canny(binary,100,200)
    #dilation = cv2.dilate(edges,np.ones((3,3),np.uint8),iterations = 1)
    dilation = cv2.dilate(edges,np.ones((5,5),np.uint8),iterations = 1)
    qr_contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

    QR_Codes = []
    for contour in qr_contours:
        QR_Codes.append(QR_Code(contour))

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    qr_corners = []

    for father_id,contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        width,height = rect[1]
        if width == 0 or height == 0 or width > img_width/2 or height > img_height/2:
            continue
        #this is not father
        if hierarchy[father_id][2] == -1:
            continue
        child1_id = hierarchy[father_id][2]# first child
        # this first child don't has second child
        if hierarchy[child1_id][2] == -1:
            continue
        child2_id = hierarchy[child1_id][2] #second child
        #box = np.int0(cv2.boxPoints(rect))
        contour_l = contours[father_id]
        contour_m = contours[child1_id]
        contour_s = contours[child2_id]
        poly_l = cv2.approxPolyDP(contour_l,0.03*cv2.arcLength(contour_l,True),True)
        poly_m = cv2.approxPolyDP(contour_m,0.03*cv2.arcLength(contour_m,True),True)
        poly_s = cv2.approxPolyDP(contour_s,0.03*cv2.arcLength(contour_s,True),True)

        area_l = cv2.contourArea(poly_l)
        if area_l == 0:continue
        area_m = cv2.contourArea(poly_m)
        if area_m == 0:continue
        area_s = cv2.contourArea(poly_s)
        if area_s == 0:continue
        area_constrain = (abs(area_l/area_m - 49.0/25.0) + abs(area_m/area_s - 25.0/9.0))/2.0
        if area_constrain >= 1.5:
            continue
        finder = QR_Finder(poly_l,poly_m,poly_s)
        for code in QR_Codes:
            code.add_finder(finder)

    print(len(QR_Codes))

    for index,qr_code in enumerate(QR_Codes):
        color = random_color()
        if qr_code.valid:
            qr_code.detect_qr(binary,index)
            qr_code.draw(process1,color=color)
            qr_code.draw(src,color=color)

    cv2.imshow("process1",process1)
    cv2.imshow("src",src)

if __name__ == '__main__':
    #cap = cv2.VideoCapture(0)
    for fn in ['qr.jpg',"qr_box.jpg","qr_mat2.jpg"]:
        frame = cv2.imread(fn,1)
        #ret, frame = cap.read()
        #cv2.imshow('frame',frame)
        start = time.time()
        qr_rect = qr_code_detect(frame)
        print('time',time.time() - start)
        if cv2.waitKey(0) == 27:
            break

    #cap.release()