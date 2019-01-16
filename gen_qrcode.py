import pyqrcode
url = pyqrcode.create('1')
url.png('code.png', scale=8)
url.show()