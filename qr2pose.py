import cv2
import numpy as np
import pyzbar
from pyzbar.pyzbar import decode
import random
import time

def sw_color(color):
    return (color[1],color[2],color[0])

def n2t(array):
    return tuple(array.astype(np.int).tolist())

def putText_on_center(img,text,center,color=(255,0,0),font=cv2.FONT_HERSHEY_PLAIN,fontScale=2,thickness=2):
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    font_width = textsize[0]
    font_height = textsize[1] + 5
    offset = np.array([-font_width/2,font_height/2],dtype=np.int)
    font_lt = center + offset
    font_rb = center - offset
    cv2.rectangle(img,n2t(font_lt),n2t(font_rb),(0,0,0),-1)
    cv2.putText(img,text,n2t(center+offset),font,fontScale,color,thickness)

class QR_Finder(object):
    def __init__(self,poly_l,poly_m,poly_s):
        self.poly_l = poly_l
        self.poly_m = poly_m
        self.poly_s = poly_s

        point_sum = np.zeros(shape=(2),dtype=np.int)
        counter = 0
        for poly in [self.poly_l,self.poly_m,self.poly_s]:
            for pt in poly:
                point_sum += pt[0]
                counter += 1
        self.center = (point_sum / counter).astype(np.int)

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
    def __init__(self,contour = None):
        self.contour = contour
        self.finders = []
        self.valid = False
        self.qr_data = None

    def _other(self,id):
        if id == 0:return (1,2)
        if id == 1:return (0,2)
        if id == 2:return (0,1)

    def calc(self,grey,binary):
        if self.valid:
            # We try to guess the id of top left finder from the 3 finders
            distance1 = np.linalg.norm(self.finders[0].center - self.finders[1].center)
            distance2 = np.linalg.norm(self.finders[0].center - self.finders[2].center)
            distance3 = np.linalg.norm(self.finders[1].center - self.finders[2].center)
            if distance1 > distance2 and distance1 > distance3:
                top_left_id_try_order = [2,1,0]
            elif distance2 > distance1 and distance2 > distance3:
                top_left_id_try_order = [1,2,0]
            else:
                top_left_id_try_order = [0,1,2]

            for top_left_id in top_left_id_try_order:
                self.calc_points(binary,top_left_id)
                if self.qr_detect(grey):
                    break

    def calc_points(self,binary,top_left_id = None):
        self.finder_base = self.finders[top_left_id]
        self.finder_side1 = self.finders[self._other(top_left_id)[0]]
        self.finder_side2 = self.finders[self._other(top_left_id)[1]]

        self.center = (self.finder_side1.center + self.finder_side2.center)/2
        self.finder_oppo_guess = self.center *2 - self.finders[top_left_id].center

        #find the other two finders
        vec0 = self.finder_oppo_guess - self.finder_base.center
        vec1 = self.finder_side1.center - self.finder_base.center
        vec2 = self.finder_side2.center - self.finder_base.center
        if np.cross(vec0,vec1) > np.cross(vec0,vec2):
            self.finder_down,self.finder_right = (self.finder_side1,self.finder_side2)
        else:
            self.finder_down,self.finder_right = (self.finder_side2,self.finder_side1)

        # find the close bonding box of QR code
        self.top_left = self.finder_base.calc_corner(self.center)
        self.top_right = self.finder_right.calc_corner(self.center)
        self.bottom_left = self.finder_down.calc_corner(self.center)
        bottom_right1 = self.center * 2 - self.top_left
        bottom_right2 = (self.top_right - self.top_left) + (self.bottom_left - self.top_left) + self.top_left
        self.bottom_right_guess = (bottom_right1+bottom_right2)/2

        # Tested! this is faster than last version
        """best_criteria = 0
        best_index = 0
        for index,pt in enumerate(self.contour):
            desire_large = np.linalg.norm(pt[0]-self.top_left)
            desire_small = np.linalg.norm(pt[0]-self.bottom_right_guess)
            criteria = (desire_large - desire_small)
            if best_criteria < criteria:
                best_criteria = criteria
                best_index = index
        self.bottom_right = self.contour[best_index][0]"""

        self.bottom_right = self.bottom_right_guess
        # define the bonding box
        bonding_box = [[self.top_left],[self.top_right],[self.bottom_right],[self.bottom_left]]
        self.bonding_box = np.array(bonding_box,dtype=np.int)

        self.top_left2 = 1.1*(self.top_left-self.center) + self.center
        self.top_right2 = 1.1*(self.top_right-self.center) + self.center
        self.bottom_left2 = 1.1*(self.bottom_left-self.center) + self.center
        self.bottom_right2 = 1.1*(self.bottom_right-self.center) + self.center
        bonding_box2 = [[self.top_left2],[self.top_right2],[self.bottom_right2],[self.bottom_left2]]
        self.bonding_box2 = np.array(bonding_box2,dtype=np.int)

        return top_left_id

    def add_finder(self,finder,must_in=False):
        if must_in:
            self.finders.append(finder)
        elif cv2.pointPolygonTest(self.contour,tuple(finder.center),False) >= 0:
            self.finders.append(finder)
        self.valid = True if len(self.finders) == 3 else False            


    def draw(self,image,color, thickness = 1):
        for poly in self.finders:
            poly.draw(image,color = color, thickness = 1)
        if self.valid:
            cv2.drawContours(image,[self.bonding_box],0,color,1)
            cv2.drawContours(image,[self.bonding_box2],0,sw_color(color),1)
            cv2.circle(image,n2t(self.center),5,(255,255,255),-1)

            cv2.circle(image,n2t(self.finder_base.center),5,(255,255,255),-1)
            cv2.circle(image,n2t(self.finder_down.center),5,(0,255,255),-1)
            cv2.circle(image,n2t(self.finder_right.center),5,(255,0,255),-1)

            cv2.circle(image,n2t(self.top_left),5,(255,255,255),-1)
            cv2.circle(image,n2t(self.bottom_left),5,(0,255,255),-1)
            cv2.circle(image,n2t(self.top_right),5,(255,0,255),-1)
            cv2.circle(image,n2t(self.bottom_right_guess),5,(255,0,0),2)
            cv2.circle(image,n2t(self.bottom_right),5,(255,255,0),-1)

            cv2.line(image,n2t(self.finder_base.center),n2t(self.finder_side1.center),color,1)
            cv2.line(image,n2t(self.finder_base.center),n2t(self.finder_side2.center),color,1)
            cv2.line(image,n2t(self.finder_side1.center),n2t(self.finder_side2.center),color,1)

    def qr_detect(self,image):
        # try to recognize the qrcode using different warp size
        for warp_size in range(100,251,50):
            pts1 = np.float32([self.top_left2,self.top_right2,self.bottom_left2,self.bottom_right2])
            pts2 = np.float32([[0,0],[0,warp_size],[warp_size,0],[warp_size,warp_size]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            self.warpPerspective = cv2.warpPerspective(image,M,(warp_size,warp_size))
            qr = decode(self.warpPerspective)
            if qr != []:
                self.qr_data = qr[0].data.decode('ascii')
                return True
        return False

def qr_code_detect(src):
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img_width,img_height = gray.shape

    process1 = np.zeros(shape=src.shape,dtype=src.dtype)

    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,203,-17)
    
    finder_contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    qr_finders=[]
    # find type1_finder -> 3 square finder
    for father_id,contour in enumerate(finder_contours):
        if cv2.arcLength(contour,True) <= 20:continue

        rect = cv2.minAreaRect(contour)
        width,height = rect[1]
        if width == 0 or height == 0 or width > img_width/2 or height > img_height/2:continue

        #this is not father
        if hierarchy[father_id][2] == -1:continue

        child1_id = hierarchy[father_id][2]# first child
        # this first child don't has second child
        if hierarchy[child1_id][2] == -1:continue

        child2_id = hierarchy[child1_id][2] #second child
        contour_l = finder_contours[father_id]
        contour_m = finder_contours[child1_id]
        contour_s = finder_contours[child2_id]

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
        qr_finders.append(finder)

    #print('->>',len(qr_finders))

    QR_Codes = []
    if len(qr_finders)<=2:
        pass
    elif len(qr_finders) == 3:
        qr_code = QR_Code()
        qr_code.add_finder(qr_finders[0],must_in=True)
        qr_code.add_finder(qr_finders[1],must_in=True)
        qr_code.add_finder(qr_finders[2],must_in=True)
        QR_Codes.append(qr_code)
    else:
        canny = cv2.Canny(binary,100,200)
        dilate = cv2.dilate(canny,np.ones((5,5),np.uint8),iterations = 3)
        qr_contours, hierarchy = cv2.findContours(dilate,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
        # find the course area of the qrcode
        for contour in qr_contours:
            length = cv2.arcLength(contour,True)
            if length <= 100 or length >= 2000:
                continue
            qr_code = QR_Code(contour)
            for finder in qr_finders:
                qr_code.add_finder(finder)
            QR_Codes.append(qr_code)

    success = 0
    for qr_code in QR_Codes:
        qr_code.calc(gray,binary)
        if qr_code.qr_data != None:
            success += 1
    return success
    
    color = (100,255,0)
    color_err = (0,0,255)
    for finder in qr_finders:
        finder.draw(src,color=color_err)
        finder.draw(process1,color=color_err)
    
    for qr_code in QR_Codes:
        font = cv2.FONT_HERSHEY_PLAIN
        if qr_code.qr_data != None:
            qr_code.draw(src,color=color)
            qr_code.draw(process1,color=color)
            putText_on_center(src,qr_code.qr_data,qr_code.center,(255,255,255))
            putText_on_center(process1,qr_code.qr_data,qr_code.center,(255,255,255))
        else:
            print('fail')
            qr_code.draw(src,color=color_err)
            qr_code.draw(process1,color=color_err)

    cv2.imshow("process1",process1)
    cv2.imshow("src",src)
    return success

def job():
    #cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture('WIN_20190117_14_21_00_Pro.mp4')
    #while True:
    for fn in ['mat0.jpg','mat1.jpg','qr_map7x5.png']:
    #for fn in ['diff1.jpg','fail4.jpg','fail1.jpg','fail2.jpg','fail3.jpg']:
        frame = cv2.imread('./test_img/'+fn,1)
        #ret, frame = cam.read()
        success = qr_code_detect(frame.copy())
        #if success == 0:
        #    cv2.imwrite('fail.jpg',frame)
        #print(success)
        #if 27 == cv2.waitKey(0):
        #    break
    #cam.release()

if __name__ == '__main__':
    import cProfile
    cProfile.run('for i in range(0,10):job()',filename='profile.cprof')
    #job()
