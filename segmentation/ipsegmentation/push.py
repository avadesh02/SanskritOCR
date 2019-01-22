## contains code to segment image into words and lines and save them in separate folders
## author = Parth Batra
## date = 13-01-2019

# IMPORTS
import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im
from scipy.ndimage import interpolation as inter

#initialising class "segment"
class segment:
    def __init__(self,image):
        self.image_show = cv2.imshow("Original Image",image)

    # different parameters calculation of image
    def image_para_calc(self, image):
        print("This image is ", type(image), "with dimensions", image.shape,"\n")

    # attempt to remove background
    # TODO: TRY TO ELIMINATE THEM
    def bg_filter(self, image,):
        color_select = np.copy(image)

        # defining color criteria
        red_threshold = 200
        green_threshold = 200
        blue_threshold = 200
        rgb_threshold = [red_threshold, green_threshold, blue_threshold]

        # identify pixels above threshold
        thresholds = (image[:, :, 0] > rgb_threshold[0]) | (image[:, :, 1] > rgb_threshold[1]) | (image[:, :, 2] > rgb_threshold[2])
        color_select[thresholds] = [255, 255, 255]

        # display image
        cv2.imshow("Background Removed", color_select)

        #converting the plot into image and returning it for further calls
        return cv2.cvtColor(np.array(color_select), cv2.COLOR_RGB2BGR)

    # conversion to binary image
    def img2binary(self, image):
        color_select = np.copy(image)
        (thresh, im_bw) = cv2.threshold(color_select, 128, 255, cv2.THRESH_BINARY)

        #cv2.imwrite("bw2.jpg", im_bw)

        shapemask = cv2.inRange(im_bw, 0, 10)
        #cv2.imwrite("mask2.jpg", shapemask)

        cv2.imshow("Binary Image", im_bw)
        #cv2.imshow("Inverse of Binary we made", shapemask)
        cv2.waitKey(0)
        return cv2.cvtColor(np.array(im_bw), cv2.COLOR_RGB2BGR)
        #return im_bw

    #detect and fix the skew of the image
    def skew_fix(self, image):
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

        # correct skew
        data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
        img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
        img.save('skew_corrected2.png')
        plt.imshow(img)
        plt.show()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # converting the processed text into separate lines
    def img2line(self, image, ori):
        k = 0
        p = 0

        ori = cv2.resize(ori, (0, 0), fx=1.69, fy=1.69)

        # TWEAK RESIZING FACTOR FOR SPACING
        image = cv2.resize(image, (0, 0), fx=1.69, fy=1.69)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # (2) Threshold
        # USABLE FOR BOTH TYPE OF BINARY IMAGES
        # for white BG and  Black text
        th, threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # for black BG and White Text
        # th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # (3) minAreaRect on the nozeros
        pts = cv2.findNonZero(threshed)
        ret = cv2.minAreaRect(pts)

        (cx, cy), (w, h), ang = ret
        if w < h:
            w, h = h, w
            ang += 90

        # (4) Find rotated matrix, do rotation
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rotated = cv2.warpAffine(threshed, M, (image.shape[1], image.shape[0]))

        # (5) find and draw the upper and lower boundary of each lines
        hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

        # TWEAK "TH" -  THRESHOLD
        th = 2
        H, W = image.shape[:2]
        uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
        lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]

        # To eliminate the small gaps
        # initiations
        upper = []
        lower = []
        diff = []
        heights = []
        l = len(uppers)

        for j in range(0, l):
            if lowers[j] - uppers[j] >= 10:
                upper.append(uppers[j])
                lower.append(lowers[j])

        for k in range(0, len(upper)):
            diff.append(lower[k] - upper[k])
            heights.append(upper[k])

        print("diff:", diff)
        print("heights:", heights)
        # To fix the multiple contours joining factor:
        # Normalise "diff" array
        minim = min(diff)

        for i in range(0, len(uppers)-1):
            diff[i] = math.floor(round(diff[i] / minim, ndigits=3))

        print("diff normalised:", diff, "\n")
        print("lowers:", lowers)
        print("lower:", lower, "\n")
        print("uppers:", uppers)
        print("upper:", upper)

        rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

        for y in upper:
            cv2.line(rotated, (0, y), (W, y), (255, 0, 0), 1)

        for y in lower:
            cv2.line(rotated, (0, y), (W, y), (0, 255, 0), 1)

        cv2.imshow("result.png", rotated)
        cv2.waitKey(0)

        def line2words(image):
            #count = 0
            #image = cv2.resize(image, (0, 0), fx=1.3, fy=1.3)

            # convert to 'grayscale' image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # smooth the image to avoid noises
            gray = cv2.medianBlur(gray, 5)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # apply some dilation and erosion to join the gaps
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.erode(thresh, None, iterations=3)

            # Find the contours
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            co_array = []
            area = []

            total_area = image.shape[0] * image.shape[1]

            # For each contour, find the bounding rectangle and draw it
            # cropping and saving to another dir ./words
            # loop for each word
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                if w * h >= (0.008 * total_area):
                    area.append((w * h))
                    co_array.append((x, y, w, h))
                    # cropping image to rectangles
                    cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # increment sample words
                #count += 1
                # saving into words folder
                ##FOR SEPARATE PAGES WE CAN EDIT NOMENCLATURE BY INTRODUCING ANOTHER STRING OF PAGE NO. LIKE [+str(text_image_no.)]
                # cv2.imwrite("words/2."+ str(count) +".jpg", image[y:y+h, x:x+w])
                print(i)
                display_words(ori,co_array,heights,i)

            print("Areas:", area)
            print("(x,y,w,h):", co_array)

            # Finally show the image
            cv2.imshow('image', image)
            cv2.imshow('res', thresh_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #display words on original image
        def display_words(ori, co_array, heights,i):
                for j in range(0,len(co_array)):
                    cv2.rectangle(ori, (co_array[j][0], heights[i] + co_array[j][1]),(co_array[j][0] + co_array[j][2], heights[i] + co_array[j][1] + co_array[j][3]),(0, 255, 0), 2)
                i = i + 1
                cv2.imshow("ori",ori)
        # for viewing and separating images

        for i in range(0, l - 1):
                cv2.imshow("line_sample" + str(i+1), image[upper[i]: lower[i], :])
                cv2.waitKey(0)
                sample_image =cv2.cvtColor(np.array(image[upper[i]:lower[i],:]), cv2.COLOR_RGB2BGR)
                line2words(sample_image)
                #cv2.imwrite("./lines/Line" + str(i+1)+".jpg", image[upper[i]:lower[i], :])
