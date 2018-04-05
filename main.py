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
        length = np.sqrt( ( x1 - x2 ) ** 2 + ( y1 - y2 ) ** 2 )
        if length > thLength:
            lineSegs.append([x1, y1, x2, y2])

    return lineSegs

def drawClusters(image, lines, clusters):
    palattee = [(255,0,0), (0,255,0), (0,0,255)]
    colorID = 0
    for cluster in clusters:
        for line_id in cluster:
            pt1 = (np.int(lines[line_id][0]), np.int(lines[line_id][1]))
            pt2 = (np.int(lines[line_id][2]), np.int(lines[line_id][3]))
            cv2.line(image, pt1, pt2, palattee[colorID], 2)
        colorID += 1

    return image

def drawVps(image, vps, pp, f):

    vp2D = [[] for i in range(3)]
    for i in range(3):
        vp2D[i] = [vps[i][0] * f / vps[i][2] + pp[0], vps[i][1] * f / vps[i][2] + pp[1]]

    pts = [[] for i in range(3)]
    for i in range(3):
        pts[i] = (np.int(vp2D[i][0]), np.int(vp2D[i][1]))

    cv2.line(image, pts[0], pts[1], (255, 255, 0), 2)
    cv2.line(image, pts[1], pts[2], (255, 255, 0), 2)
    cv2.line(image, pts[2], pts[0], (255, 255, 0), 2)

    return image

if __name__ == '__main__':
    import argparse
    import cv2
    import numpy as np
    from pylsd.lsd import lsd
    from lib import VPDetection

    parser = argparse.ArgumentParser(description="Vanishing point detection script with camera intrinsic parameter decision.")
    parser.add_argument("-f", "--file", dest = "filename", type=str, metavar="FILE", help = "Give the address of image source")
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

    f = np.max(image.shape)# focal length (in pixel), a former guess

    noiseRatio = 0.5
    # VPDetection class
    detector = VPDetection(lines, pp, f, noiseRatio)
    vps, clusters = detector.run()

    image1 = drawClusters(image, lines, clusters)

    cv2.imshow("", image1)
    cv2.waitKey(0)




