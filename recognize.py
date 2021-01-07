from ANPR.license_plate.licenseplate import LicensePlateDetector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", required=True, help="path to the images")
args = vars(parser.parse_args())

# loop over the images
for imagePath in sorted(list(paths.list_images(args["images"]))):
    image = cv2.imread(imagePath)
    print(imagePath)

    # if the width is greater than 640 pixels, then resize the image
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    # initilaize the license plate detector and detect the license plates and characters
    lpd = LicensePlateDetector(image)
    plates = lpd.detect()

    # loop over the detected plates
    for (lpBox, chars) in plates:
        # loop over each character
        for (i, char) in enumerate(chars):
            # show the charater
            cv2.imshow("Character {}".format(i + 1), char)
    
    # display the output image
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()