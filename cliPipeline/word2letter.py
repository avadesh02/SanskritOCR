
# coding: utf-8

# In[12]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# In[13]:


def letterSegmentation(img):

    img = cv2.resize(img,(0,0),fx = 1.69, fy = 1.69)

#     print (img.shape)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    threshed_color = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR)

    intensity = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
#     print(np.max(intensity))

    maxRows = [i for i in range(0,len(intensity)) if intensity[i] == np.max(intensity)]
    pixels = 13
    med = np.median(maxRows)
    (a,b) = (med - (pixels-1)/2,med + (pixels-1)/2)
#     print(a,b)

    # removing header line
    for i in range(math.floor(a),math.ceil(b)):
        threshed[i] = np.array(0)
    
#     cv2.imshow("im",threshed)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    threshed = cv2.rotate(threshed,cv2.ROTATE_90_CLOCKWISE)
    threshed_color = cv2.rotate(threshed_color,cv2.ROTATE_90_CLOCKWISE)

    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
    #plt.plot(hist)

    th = 2
    H, W = threshed.shape
    uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
    lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]

    upper = []
    lower = []
    diff = []
    lefts = []
    l = len(uppers)

    for j in range(0, l):
        if lowers[j] - uppers[j] >= 5:
            upper.append(uppers[j])
            lower.append(lowers[j])

    for k in range(0, len(upper)):
        diff.append(lower[k] - upper[k])
        lefts.append(upper[k])

#   print("diff:", diff)
#    print("lefts:", lefts)
    # To fix the multiple contours joining factor:
    # Normalise "diff" array
    minim = sum(diff)/len(diff)
#     print(minim)
    for i in range(0, len(diff)):
        diff[i] = (round(diff[i] / minim, ndigits=3))

#     print("diff normalised:", diff, "\n")
# #    print("lowers:", lowers)
#     print("lower:", lower, "\n")
# #    print("uppers:", uppers)
#     print("upper:", upper)

#     for y in upper:
#         cv2.line(threshed_color, (0, y), (W, y), (255, 0, 0), 1)

#     for y in lower:
#         cv2.line(threshed_color, (0, y), (W, y), (0, 255, 0), 1)

    
    final_img = np.zeros((25,threshed_color.shape[1]))    
    for i in range(0,len(upper)):
        final_img = np.concatenate((final_img,threshed[upper[i]:lower[i]][:]))
        final_img = np.concatenate((final_img,np.zeros((50,threshed_color.shape[1]))))
        

    ls = []
    for i in range(0,len(upper)):
        #cv2.imwrite("./ltr/"+str(n)+"_"+str(upper[i])+"_"+str(lower[i])+".jpg",cv2.rotate(threshed[upper[i]:lower[i]][:],cv2.ROTATE_90_COUNTERCLOCKWISE))
        ls.append(cv2.rotate(threshed_color[upper[i]:lower[i]][:],cv2.ROTATE_90_COUNTERCLOCKWISE))
    
    threshed = cv2.rotate(threshed,cv2.ROTATE_90_COUNTERCLOCKWISE)
    threshed_color = cv2.rotate(threshed_color,cv2.ROTATE_90_COUNTERCLOCKWISE)
    final_img = cv2.rotate(final_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cv2.imshow("result.png", final_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print(final_img.shape)
    return ls


# In[14]:


ls = np.load("words.npy")
segmented_words = []
for img in ls:
	segmented_words.append(letterSegmentation(img))


# resizing the image to (32,32,3)
seg_word = []

for word in segmented_words:
	wd = []
	for img in word:
	    factor = min(32/img.shape[0],32/img.shape[1])

	    img = cv2.resize(img,None,fx = factor, fy = factor, interpolation=cv2.INTER_AREA)

	    if (img.shape[0] == 32):
	        x = 16-(img.shape[1]/2)
	        y = 0
	    elif(img.shape[1] == 32):
	        x=0
	        y=16 - (img.shape[0]/2)

	    M = np.float32([[1,0,x],[0,1,y]])
	    shrinked = cv2.warpAffine(img,M,(32,32))
	    wd = wd + [shrinked]
	    cv2.imshow("res",shrinked)
	    cv2.waitKey(500)
	    cv2.destroyAllWindows()
	seg_word = seg_word + [wd]
seg_word = np.asarray(seg_word)


np.save("letters",seg_word)
# In[15]:


'''
first is list with dim equal to no.of words, 
second is a list of segmented letters, 
and each letter again is a 3 dim(32,32,3) image of "segmented_words"
'''








