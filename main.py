def LineDetect(image, thLength):
    if image.shape[2] == 1:
        grayImage = image
    else:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imageLSD = np.copy(grayImage)

    # line segments, [pt1[0], pt1[1], pt2[0], pt2[1], width]
    linesLSD = lsd(imageLSD)
    del imageLSD

    # choose line segments whose length is less than thLength
    lineSegs = []
    for line in linesLSD:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        length = np.sqrt( ( x1 - x2 ) * ( x1 - x2 ) + ( y1 - y2 ) * ( y1 - y2 ) )
        if length > thLength:
            lineSegs.append([x1, y1, x2, y2])

    return lineSegs

if __name__ == '__main__':
    import argparse
    import cv2
    import numpy as np
    from pylsd.lsd import lsd
    from lib import VPDetection

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

    noiseRatio = 0.5
    # VPDetection class
    detector = VPDetection(lines, pp, f, noiseRatio)
    lines, clusters = detector.run()

    drawClusters(image, lines, clusters)
    imshow("Image", image)
    cv2.waitKey(0)




