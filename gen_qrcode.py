import pyqrcode
import cv2
import numpy as np
for i in range(7):
    for j in range(5):
        qr_code = pyqrcode.create('%d-%d' % (i,j))
        qr_code.png('./qr_code/%d-%d.png' % (i,j), scale=8)


col = np.zeros(shape=(0))
for i in range(7):
    row = np.zeros(shape=(0))
    for j in range(5):
        qr = cv2.imread('./qr_code/%d-%d.png'%(i,j),0)
        if row.shape[0] == 0:
            row = qr
        else:
            row = np.vstack((row, qr))
    if col.shape[0] == 0:
        col = row
    else:
        col = np.hstack((col, row))
cv2.imwrite('qr_map7x5.png',col)
        