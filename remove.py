import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('img/bb.jpg')

original=img.copy()

l = int(max(5, 6))
u = int(min(6, 6))

edges = cv.GaussianBlur(img, (21, 51),3)
edges = cv.cvtColor(edges , cv.COLOR_BGR2GRAY)


edges = cv.Canny(edges,l,u)
# edges = cv.dilate(edges, None)
# edges = cv.erode(edges, None)

_,thresh=cv.threshold(edges,0,255,cv.THRESH_BINARY  + cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)


# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30,30))
# morphed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
# dilate=cv.dilate(morphed, None, iterations = 30) #change iteration

# mask = dilate

# (cnt, _) = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# for contour in cnt:
#     if len(contour) > 40:
#         cv.drawContours(mask ,contour, -1, (255, 255, 255), 50)
#         continue


# cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=cv.contourArea, reverse=True)
# for c in cnts:
#     cv.drawContours(mask, [c], -1, (255,255,255), 30)
#     break

# # msk=mask
# # mask=255-mask
# mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=4)
# # mask=255-mask


# result = cv.bitwise_and(original, original, mask=mask)
# result[mask==0] = (0,0,0)

img1=cv.resize(edges,(600,400))
import sys
img=cv.resize(mask,(300,200))
data=img.tolist()
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
plt.imshow(image)

# plt.imshow(result,cmap = 'gray')
# plt.title('edge')
# plt.show()

