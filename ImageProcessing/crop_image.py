import cv2
import imutils


def crop_image(image):
    """
    function to crop images
    :param image: image after mpimg.imread
    :return:
    """
    # convert to 8 bit image
    image_8bit = cv2.convertScaleAbs(image)

    gray = image_8bit

    # convert the image to grayscale
    if len(image_8bit.shape) == 3:
        if image_8bit.shape[2] != 1:
            gray = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2GRAY)

    # apply a Gaussian blur to the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image by Binary Thresholding
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    # perform a series of erosion's & dilation's to remove any small regions of noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in threshold image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea, default=0)

    # find the extreme points
    # if they are 0 return the original image
    if not isinstance(c, int):
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        # crop
        new_img = image_8bit[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()

        return new_img
    else:
        return image_8bit
