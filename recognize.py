from ANPR.license_plate.licenseplate import LicensePlateDetector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import pickle
from ANPR.descriptors.blockbinarypixelsum import BlockBinaryPixelSum

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", required=True, help="path to the images")
parser.add_argument("-c", "--char-classifier", required=True,
	help="path to the output character classifier")
parser.add_argument("-d", "--digit-classifier", required=True,
	help="path to the output digit classifier")
args = vars(parser.parse_args())

# load the character and digit classifiers
charModel = pickle.loads(open(args["char_classifier"], "rb").read())
digitModel = pickle.loads(open(args["digit_classifier"], "rb").read())
 
# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

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
        # restructure lpBox
        lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

        # initialize the text containing the recognized characters
        text = ""

        # loop over each character
        for (i, char) in enumerate(chars):
            # pre-process the character and describe it
            char = LicensePlateDetector.preprocessChar(char)
            if char is None:
                continue
            features = desc.describe(char).reshape(1, -1)

            # if this is the first 3 characters, then use the character classifier
            if i < 3:
                prediction = charModel.predict(features)[0]

            # otherwise, use the digit classifier
            else:
                prediction = digitModel.predict(features)[0]

            # update the text of the recognized characters
            text += prediction.upper()
        
        # only draw the characters and bounding box if there are some characters that
        # we can display
        if len(chars) > 0:
            # compute the center of the license plate bounding box
            M = cv2.moments(lpBox)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the license plate region and license plate text on the image
            cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
            cv2.putText(image, text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2)

    # display the output image
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()