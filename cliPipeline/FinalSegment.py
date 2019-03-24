
# coding: utf-8

# # Importing Libraries

# In[1]:


import cv2
import math
import numpy as np
import matplotlib
import matplotlib.image as mpimg
from PIL import Image as im
from scipy.ndimage import interpolation as inter
from scipy import stats
import os
import shutil
#import matplotlib.pyplot as plt
import sys


# # Code for Segementation to word Level

# In[10]:


# Function to remove background texture --- Will work if texture is comparatively light as compared to text
def bg_filter(image):
	color_select = np.copy(image)

	# defining color criteria
	#TODO -- to tweak this parameter for efficient backgorund clear
	val = 200
	red_threshold = val
	green_threshold = val
	blue_threshold = val
	rgb_threshold = [red_threshold, green_threshold, blue_threshold]

	# identify pixels above threshold
	thresholds = (image[:, :, 0] > rgb_threshold[0]) | (image[:, :, 1] > rgb_threshold[1]) | (image[:, :, 2] > rgb_threshold[2])
	color_select[thresholds] = [255, 255, 255]

	return cv2.cvtColor(np.array(color_select), cv2.COLOR_RGB2BGR)

# Function to pad image with a paticular pixel value
def pad_image(img,val):
	x = 2
	y = 2
	w,h = img.shape[0],img.shape[1]
	M = np.float32([[1, 0, x], [0, 1, y]])
	padded = cv2.warpAffine(img, M, (h + 4, w + 4), borderValue=(val, val, val))
	return padded

# Function to create binary image
def img2binary(image):
	color_select = np.copy(image)
	(thresh, im_bw) = cv2.threshold(color_select, 128, 255, cv2.THRESH_BINARY_INV)

	return cv2.cvtColor(np.array(im_bw), cv2.COLOR_RGB2BGR)

# Function to fix skew angle if any?
def skew_fix(image):
	# convert to binary
	image = im.fromarray(image)
	wd, ht = image.size
	pix = np.array(image.convert('1').getdata(), np.uint8)
	bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

	def find_score(arr, angle):
		data = inter.rotate(arr, angle, reshape=False, order=0)
		hist = np.sum(data, axis=1)
		score = np.sum((hist[1:] - hist[:-1]) ** 2)
		return hist, score

	delta = 0.5
	limit = 7
	angles = np.arange(-limit, limit + delta, delta)
	scores = []
	for angle in angles:
		hist, score = find_score(bin_img, angle)
		scores.append(score)

	best_score = max(scores)
	best_angle = angles[scores.index(best_score)]
	print("Best angle for skew correction:", best_angle)
	print()
	# correct skew
	data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
	img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
	#         img.save('skew_corrected2.png')
	#         plt.imshow(img)
	#         plt.show()
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Function to seperate lines and words 
def continuousSubsequence(x,th,diff):
	up = []
	down =[]
	i = 0
	while(i<len(x)-1):
		if(x[i] > th):
		    up.append(i)
	#             print("up: " +str(i),end='\t')
		    i = i+1
		    while(not(x[i] <= th) and i<len(x)-1):
		        i = i+1
		    down.append(i)
	#             print("down: " +str(i))
		    i = i+1
		else:
		    i = i+1
	u = []
	d = []
	for i in range(0,len(up)):
		if(down[i]-up[i]>diff):
		    u.append(up[i])
		    d.append(down[i])
	return u,d

# Main function to sepearte lines and words from image
def img2line(image):

	# TWEAK RESIZING FACTOR FOR SPACING
	image = cv2.resize(image, (0, 0), fx=1.69, fy=1.69)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	H,W = image.shape[0],image.shape[1]

	# coverting image to BW
	th, rotated = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	rotated = cv2.dilate(rotated, None, iterations=3)
	rotated = cv2.erode(rotated, None, iterations=2)
	rotated = pad_image(rotated,0)

	# seperating to lines using histogram approach
	hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)
	#     plt.plot(hist)
	#     print(hist)
	mode = min(stats.mode(hist)[0])
	#     print(mode)
	upper,lower = continuousSubsequence(hist,mode,10)

	#     print("uppers:", uppers)
	#     print("lowers:", lowers)

	diff = []
	for k in range(0, len(upper)):
		diff.append(lower[k] - upper[k])

	def nearestInt(x):
		f,i = math.modf(x)
		if(f<.6):
		    return i
		else:
		    return i+1

	print("diff:", diff)
	minim = min(diff)
	for i in range(0, len(diff)):
		diff[i] = int(nearestInt(diff[i] / minim))
		
	print("diff normalised:", diff, "\n")
	#     print("lower:", lower, "\n")
	#     print("upper:", upper)

	def breakImg(up,low,n,points):
		if(n==1):
		    return points
		else:
		    points = points + [int(((n-1)*up + low)/n)]
		    return breakImg(int(((n-1)*up + low)/n),low,n-1,points)

	up = []
	low = []
	for i in range(0,len(diff)):
		if(diff[i] > 1):
		    points = breakImg(upper[i],lower[i],diff[i],[])
		    up = up + [upper[i]]
		    for j in points:
		        up = up+[j]
		        low = low + [j];
		    low = low + [lower[i]]
		else:
		    up.append(upper[i])
		    low.append(lower[i])

	print("up:", up)
	print("low:", low)


	rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
	for y in up:
		cv2.line(rotated, (0, y), (rotated.shape[1], y), (255, 0, 0), 1)

	for y in low:
		cv2.line(rotated, (0, y), (rotated.shape[1], y), (0, 255, 0), 1)

	cv2.imshow("result.png", rotated)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	def line2words(image,up,down,H,W):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# smooth the image to avoid noises
		gray = cv2.medianBlur(gray, 5)

		# Apply adaptive threshold
		thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
		thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

		# apply some dilation and erosion to join the gaps
		thresh = cv2.dilate(thresh, None, iterations=3)
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = pad_image(thresh,0)
		
		hist = cv2.reduce(thresh, 0, cv2.REDUCE_AVG).reshape(-1)
	#         plt.plot(hist)
	#         print(hist)
		#TODO -- Need to tweak threshhold parameter
		lefts,rights = continuousSubsequence(hist,5,15)

		margin = 3
		ls = []
		for i in range(0,len(lefts)):
		    temp = (max(up-margin,0),min(down+margin,H-1),max(lefts[i]-margin-3,0),min(rights[i]+margin,W-1))
		    ls.append(temp)
	#             print(temp)
		
		return ls

	word_list = []
	for i in range(0, len(up)):
		sample_image =cv2.cvtColor(np.array(image[up[i]:low[i],:]), cv2.COLOR_RGB2BGR)
		word_list = word_list+line2words(sample_image,up[i],low[i],H,W)
		
	return word_list


# # Driver code for getting words

# In[11]:

print("initiated")
img = cv2.imread(sys.argv[1])
ori = np.copy(img)
img = bg_filter(img)
img = img2binary(img)
img = skew_fix(img)
result = img2line(img)
# cv2.imshow("res",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Output of Words

# In[12]:


ori = cv2.resize(ori,(0,0),fx = 1.69,fy = 1.69)


# In[13]:


copy = np.copy(ori)
for point in result:
    cv2.rectangle(copy,(point[2],point[0]),(point[3],point[1]),(0,255,0),2)
cv2.imshow('res', copy)
cv2.waitKey(0)
cv2.destroyAllWindows()  
#np.save("words",result)


# # Creating Words array

# In[14]:

img = cv2.resize(img,(0,0),fx = 1.69,fy = 1.69)
words = [(img[point[0]:point[1],point[2]:point[3]],np.copy(ori[point[0]:point[1],point[2]:point[3]])) for point in result]
words = [(pad_image(img[0],255),pad_image(img[1],255))for img in words]
words = np.asarray(words)


# In[16]:


# for img in words:
#     cv2.imshow("bw",img[0])
#     cv2.imshow("ori",img[1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# # Letter Level Segmentation
# #### Author: Anupam Aggarwal

# In[17]:


fSize = [0,0]


# In[18]:


def binaryImages(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    threshed_color = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR)
    return gray,threshed,threshed_color

# Function to remove header line -- May Fail if intesity of points is max at points other than at header line
def rmHeaderLine(threshed):
    intensity = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)    
#     plt.plot(intensity)
#     print(intensity)
    maxRows = [i for i in range(0,len(intensity)) if intensity[i] == np.max(intensity)]
    pixels = 13
    med = maxRows[len(maxRows)//2] if (len(maxRows)%2 == 1) else (maxRows[len(maxRows)//2] + maxRows[len(maxRows)//2-1])/2
    (a,b) = (med - (pixels-1)/2,med + (pixels-1)/2)
    
#     print(max(math.floor(a),0),min(math.ceil(b),len(intensity)))
    # removing header line
    for i in range(max(math.floor(a),0),min(math.ceil(b),len(intensity))):
        threshed[i] = np.array(0)
    
    return (a,b),threshed
            

def verticalSeperation(threshed,th):
    #threshed = pad_image(threshed)
    threshed = cv2.rotate(threshed,cv2.ROTATE_90_CLOCKWISE)

    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
#     print(hist)
#     plt.plot(hist)
	#TODO -- tweaking of th Parameter
    th = 2
    upper,lower = continuousSubsequence(hist,th,5)
    
    return upper,lower


def resize2_32(img):
    maxInd = 0 if (img.shape[0] > img.shape[1]) else 1
    fac = 32/img.shape[maxInd]
    img = cv2.resize(img,(0,0),fx=fac,fy=fac)
    if(img.shape[maxInd] != 32):
        newSize = (32,img.shape[1]) if maxInd==0 else (img.shape[0],32)
        img = cv2.resize(img,newSize)
    delta_w = 32 - img.shape[1]
    delta_h = 32 - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    if(left!=0 and right != 0):
        tmp = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        border = tmp[0]
        border = np.expand_dims(border,axis=1)
        img = np.concatenate((border,img),axis=1)
        img = np.concatenate((img,border),axis=1)

    elif(top!=0 and bottom != 0) :
        tmp = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        border = tmp[:,0]
        border = np.expand_dims(border,axis=0)
        img = np.concatenate((border,img),axis=0)
        img = np.concatenate((img,border),axis=0)
#     print(img.shape)
    delta_w = 32 - img.shape[1]
    delta_h = 32 - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
#     print(top,bottom,left,right)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

    return img

def makeCollage(ls):
    length = 0
    if(type(ls)=='np.ndarray'):
        length = ls.shape[0]
    else:
        length = len(ls)
    
    col = math.floor(math.sqrt(length))
    row = length//col
    #print (row,col)
    
    res = ls[0]
    for i in range(1,col):
        res = np.concatenate((res,ls[i]),axis=1)
      
    for i in range(1,row):
        temp = ls[i*col]
        for j in range(1,col):
            temp = np.concatenate((temp,ls[i*col+j]),axis=1)
        res = np.concatenate((res,temp))
        
    rem = length - row*col
    if(rem>0):
        temp = ls[row*col]
        for i in range(1,rem):
            temp = np.concatenate((temp,ls[row*col+i]),axis=1)
        
        for j in range(rem,col):
            tp = np.zeros(ls[1].shape,dtype="uint8")
            temp = np.concatenate((temp,tp),axis=1)
        
        res = np.concatenate((res,temp))
    
    return res

def determineFsize(img):
    global fSize
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img,kernel,iterations=2)
    img = cv2.blur(img,(3,3))
    img = cv2.erode(img,kernel,iterations=2)
    _,contours,_ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (h > 20):
            fSize = [fSize[0]+1,(fSize[0]*fSize[1] + h)/(fSize[0]+1)]
#             cv2.imshow("res",cv2.rectangle(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),(x,y),(x+w,y+h),(0,255,0),1))
#             cv2.waitKey(500)
#             cv2.destroyAllWindows()

def firstlevelSegment(img):
    img = cv2.resize(img,(0,0),fx = 1.69, fy = 1.69)
    gray, threshed, threshed_color = binaryImages(img)

    # removing header line
    (u,l),threshed = rmHeaderLine(threshed)
    
    # vertically seperating
    lefts,rights = verticalSeperation(threshed,2)
    
    for i in range(0,len(lefts)):
        determineFsize(threshed[int(l):img.shape[0],lefts[i]:rights[i]])
       
    return (u,l),lefts,rights

def thirdLevelSegmentation(img,head,lt,rt):
    img0 = cv2.resize(img[0],(0,0),fx=1.69,fy=1.69)
    img1 = cv2.resize(img[1],(0,0),fx=1.69,fy=1.69)
#     print("BW "+str(img[0].shape)+" ORI "+str(img[1].shape))
#     print()
 
    _, threshed, threshed_color = binaryImages(img0)
#     print("BW "+str(threshed.shape)+" ORI "+str(img[1].shape))
#     print()

    lst = []
    lst_ori = []
    for i in range(0,len(lt)):
        # lis = []
        letter = threshed[:,lt[i]:rt[i]]
        #letter = threshed
        up = letter[:int(head[0])]
        middle = letter[int(head[1]):int(head[1]+fSize[1]-4)]
        below = letter[int(head[1]+fSize[1]-7):]
        
        letter_ori = img1[:,lt[i]:rt[i]]
        up_ori = letter_ori[:int(head[0])]
        middle_ori = letter_ori[int(head[1]):int(head[1]+fSize[1]-4)]
        below_ori = letter_ori[int(head[1]+fSize[1]-7):]

        if((np.sum(up)/255) > 25):
            up = resize2_32(up)
            up[:3,:3] = 255*np.ones((3,3))
            lst.append(up)
            up_ori = resize2_32(up_ori)
            lst_ori.append(up_ori)

        middle = pad_image(middle,0)
        left,right = verticalSeperation(middle,20)
        for n in range(0,len(left)):
            lst.append(resize2_32(middle[:,left[n]:right[n]]))
            lst_ori.append(resize2_32(middle_ori[:,left[n]:right[n]]))

            
        if((np.sum(below)/255) > 125):
            below = resize2_32(below)
            below[-3:,:3] = 255*np.ones((3,3))
            lst.append(below)
            below_ori = resize2_32(below_ori)
            lst_ori.append(below_ori)

        
    return (lst,lst_ori)


def secondLevelSegmentation(img,up,dn):
    img = cv2.resize(img,(0,0),fx = 1.69, fy = 1.69)
    img = img[int(up):int(dn)]

    gray, threshed, threshed_color = binaryImages(img)
    lefts,rights = verticalSeperation(threshed,10)
    
    return lefts,rights


# In[19]:


def letterSegmentation(ls):
	leftsLs = []
	rightsLs = []
	headerLs = []

	# First Level Segmentation
	# getting coordinates for each letter, NECESSARY
	for i in range(0,len(ls)):
		img = ls[i][0]
		head, left, right = firstlevelSegment(img)
		leftsLs.append(left)
		rightsLs.append(right)
		headerLs.append(head)
	print(fSize)
	# Third Level Segmentation
	letters = []
	for i in range(0,len(ls)):
		img = ls[i];
		head = headerLs[i]
		lt = leftsLs[i]
		rt = rightsLs[i]
		letters.append(thirdLevelSegmentation(img,head,lt,rt))
	
	# Second Level Segmentation
	del leftsLs,rightsLs
	leftsLs = []
	rightsLs = []
	for i in range(0,len(ls)):
		img = ls[i][0]
		up = headerLs[i][1]
		dn = up + fSize[1]-4
		lt,rt = secondLevelSegmentation(img,up,dn)
		leftsLs.append(lt)
		rightsLs.append(rt)

	return leftsLs,rightsLs,headerLs,np.asarray(letters)


# In[20]:


leftsLs,rightsLs,headerLs,ls = letterSegmentation(words)


# In[21]:


# Showing Boundaries
copy = np.copy(ori)
for i in range(0,len(words)):
    point = result[i]
    img = copy[point[0]:point[1],point[2]:point[3]]
    img = cv2.resize(img,(0,0),fx=1.69,fy=1.69)
    cv2.line(img,(0,int(headerLs[i][1]+fSize[1]-4)),(img.shape[1],int(headerLs[i][1]+fSize[1]-4)),(0,255,0),3)
    cv2.line(img,(0,int((headerLs[i][0]+headerLs[i][1])/2)),(img.shape[1],int((headerLs[i][0]+headerLs[i][1])/2)),(0,255,0),3)
    for j in range(0,len(leftsLs[i])):
        cv2.line(img,(leftsLs[i][j],int((headerLs[i][0]+headerLs[i][1])/2)),(leftsLs[i][j],int(headerLs[i][1]+fSize[1]-2)),(0,0,255),2)
        cv2.line(img,(rightsLs[i][j],int((headerLs[i][0]+headerLs[i][1])/2)),(rightsLs[i][j],int(headerLs[i][1]+fSize[1]-2)),(255,0,0),2)
    img = cv2.resize(img,(point[3]-point[2],point[1]-point[0]))
    copy[point[0]:point[1],point[2]:point[3]] = img

cv2.imshow("res",copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:

# uncomment to see each segmented letter
'''
for word in ls:
    for lt in word[0]:
        if(np.array_equal(lt[:3,:3],255*np.ones((3,3)))):
            cv2.imshow('upper',lt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif(np.array_equal(lt[-3:,:3],255*np.ones((3,3)))):
            cv2.imshow('lower',lt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow('middle',lt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
'''

# In[22]:


letters_bw= []
for word in ls:
    for lt in word[0]:
        letters_bw.append(lt)

letters_ori= []
for word in ls:
    for lt in word[1]:
        letters_ori.append(lt)

res = makeCollage(letters_bw)
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

res = makeCollage(letters_ori)
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


'''
Final result which is to be predicted is Letters.
Ist dimension is Number of words in input paragraph
2nd dimension are Black and white images of letters or modifiers to predicted
'''

# FOr additional viewing purposes

# ## Resizing of letters to (32,32)

# In[ ]:


lst = []
words = [cv2.resize(ori[point[0]:point[1],point[2]:point[3]],(0,0),fx=1.69,fy=1.69) for point in result]


# In[ ]:


for i in range(0,len(words)):
    img = words[i]
    for j in range(0,len(leftsLs[i])):
        tmp = img[:,leftsLs[i][j]-2:rightsLs[i][j]+2]
        lst.append(resize2_32(tmp))

# In[ ]:


res = makeCollage(lst)
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# code to be used for prediction -- IN DEVELOPMENT
'''
# # Predicting Each Letter and Modifier

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
from keras.preprocessing import image
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
import cv2
from keras.utils import to_categorical


# In[ ]:


encoding = np.load('./Data/encodingDict600.npy').item()

json_file = open('./Data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./Data/model.h5")
print("Loaded model from disk")

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
model.summary()


# In[ ]:


for img in lst:
    img = np.expand_dims(img,axis=0)
    cl = model.predict_classes(img/255)
    cv2.imshow(encoding[str(cl[0])],img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:


# for word in ls:
#     for img in word:
#         #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#         img = np.expand_dims(img,axis=2)
#         img = np.expand_dims(img,axis=0)
#         cl = model.predict_classes(img/255)
#         cv2.imshow(encoding[str(cl[0])],img[0])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
'''
