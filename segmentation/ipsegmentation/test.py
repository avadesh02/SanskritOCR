import cv2

img = cv2.imread(r"C:\Users\Dell\Desktop\SanskritOCR\source\bw2.jpg")

image = img

img = cv2.resize(img, (image.shape[1]*5, image.shape[0]*5))

# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray,13)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 4)
thresh = cv2.erode(thresh,None,iterations = 3)

# Find the contours
_,contours, _= cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
# cropping and saving to another dir ./words
#loop for each word
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #cropping img to rectangles
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #increment sample words
    #count += 1

    #saving into words folder
    ##FOR SEPARATE PAGES WE CAN EDIT NOMENCLATURE BY INTRODUCING ANOTHER STRING OF PAGE NO. LIKE [+str(text_image_no.)]
    #cv2.imwrite("words/1."+ str(count) +".jpg", img[y:y+h, x:x+w])

    #bounding threshold image
    cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)

# Finally show the image
cv2.imshow('img',img)
cv2.imshow('res',thresh_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
