import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from PIL import Image



img = cv.imread('img/bb.jpg', cv.IMREAD_UNCHANGED)


original=img.copy()

l = int(max(5, 6))
u = int(min(6, 6))

ed = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

edges = cv.GaussianBlur(img, (21, 51),3)

edges = cv.cvtColor(edges , cv.COLOR_BGR2GRAY)

edges = cv.Canny(edges,l,u)

_,thresh=cv.threshold(edges,0,255,cv.THRESH_BINARY  + cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)






data=mask.tolist()
sys.setrecursionlimit(10**8) 
for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j]!=255:
            data[i][j]=-1
        else:
            break
    for j in range(len(data[i])-1,-1,-1):
        if data[i][j]!=255:
            data[i][j]=-1
        else:
            break
image=np.array(data)
image[image!=-1]=255
image[image==-1]=0

mask = np.array(image,np.uint8)

# mask=255-mask

result = cv.bitwise_and(original, original, mask=mask)
result[mask==0] = 255 # white background
cv.imwrite('bg.png',result)


# img=cv.resize(result,(600,400))
# cv.imshow('Original Image', img)
# cv.waitKey()

#remove white

img= Image.open('bg.png')
img.convert("RGBA")
datas=img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255: #checking white
        newData.append((255, 255, 255, 0))      #setting alpha to 0 
    else:
        newData.append(item)

img.putdata(newData)
img.save("img.png", "PNG")


 
# plt.imshow(result)
# plt.title('edge')
# plt.show()
