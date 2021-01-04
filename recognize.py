from pyimagesearch.license_plate.licenseplate import LicensePlateDetector
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

    # loop over the license plates regions an draw the bounding box surronding the 
    # license plate
    for lpbox in plates:
        lpbox = np.array(lpbox).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [lpbox], -1, (0, 255, 0), 2)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)