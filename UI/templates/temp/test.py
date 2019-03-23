import numpy as np
import cv2

def array_to_files():
    image_array = np.load('letter.npy')
    for i in range(len(image_array)):
        cv2.imwrite('letters/' + str(i) +'.png',image_array[i])

def main():
    print("This function will generate images")
    array_to_files()
    print("Images have been generated")

main()
