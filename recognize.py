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

    # loop over the license plates regions an draw the bounding box surronding the 
    # license plate
    for (i, (lp, lpbox)) in enumerate(plates):
        lpbox = np.array(lpbox).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [lpbox], -1, (0, 255, 0), 2)

        # show the output images 
        candidates = np.dstack([lp.candidates] * 3)
        thresh = np.dstack([lp.thresh] * 3)
        output = np.vstack([lp.plate, thresh, candidates])
        cv2.imshow("Plate and Candidates #{}".format(i + 1), output)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()