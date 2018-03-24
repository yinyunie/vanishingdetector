def LineDetect(image, thLength):
    return 0

if __name__ == '__main__':
    import argparse
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser(description="Vanishing points detection script.")
    parser.add_argument("-f", "--file", dest = "filename", type=str, metavar="FILE", help = "Give the address of image address")
    args = parser.parse_args()

    # Read source image
    inPutImage = args.filename

    try:
        image = cv2.imread(inPutImage)
    except IOError:
        print 'Cannot open the image file, please verify the image address.'

    # Line segment detection
    thLength = 30.0 # threshold of the length of line segments

    # detect line segments from the source image
    lines = LineDetect( image, thLength)

    # Camera internal parameters
    pp = image.shape[1]/2., image.shape[0]/2. # principle point (in pixel)
    f = np.max(image.shape)# focal length (in pixel)

    # Vanishing point detection
    detector = VPDetection()




