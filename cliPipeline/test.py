from segFunc import segment
import cv2
import sys

if (len(sys.argv)<2):
	print("Give path to image as command line argument")

# read the image and send it to segment function. It will return corresponding list with word level information
img = cv2.imread(sys.argv[1])
letters = segment(img)

for word in letters:
	for lt in word:
		cv2.imshow("res",lt)
		cv2.waitKey(500)
		cv2.destroyAllWindows()

