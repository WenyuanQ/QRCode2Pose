import pyqrcode
url = pyqrcode.create('12')
url.png('./qr_code/code.png', scale=8)
url.show()