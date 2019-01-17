import pyqrcode
import cv2
import numpy as np

def n2t(array):
    return tuple(array.astype(np.int).tolist())

def putText_on_center(img,text,center,color=(255,0,0),font=cv2.FONT_HERSHEY_PLAIN,fontScale=2,thickness=2):
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    offset = np.array([ -textsize[0] / 2,textsize[1] / 2],dtype=np.int)
    cv2.putText(img,text,n2t(center+offset),font,fontScale,color,thickness)


for i in range(7):
    for j in range(5):
        qr_code = pyqrcode.create('%d-%d' % (i,j))
        qr_code.png('./qr_code/%d-%d.png' % (i,j), scale=8)
        qr = cv2.imread('./qr_code/%d-%d.png' % (i,j))
        center = np.array([qr.shape[0]/2,qr.shape[1]*0.93])
        putText_on_center(qr,'%d-%d' % (i,j),center,(0,0,255),thickness=2)
        cv2.imwrite('./qr_code/%d-%d.png' % (i,j),qr)


col = np.zeros(shape=(0))
for i in range(7):
    row = np.zeros(shape=(0))
    for j in range(5):
        qr = cv2.imread('./qr_code/%d-%d.png'%(i,j),1)
        if row.shape[0] == 0:
            row = qr
        else:
            row = np.vstack((row, qr))
    if col.shape[0] == 0:
        col = row
    else:
        col = np.hstack((col, row))
cv2.imwrite('./qr_code/qr_map7x5.png',col)
        