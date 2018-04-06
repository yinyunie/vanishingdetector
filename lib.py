import cv2
import numpy as np

class VPDetection:

    # initiate with line segments, principal point, focal length, ratio of outliers

    def __init__(self, lines, pp, f, noiseRatio):

        self.lines = lines
        self.pp = pp
        self.f = f
        self.noiseRatio = noiseRatio
        self.lineInfos = [] # para, length, orientation

    def run(self):
        '''Main run procedure'''

        print "Get vp hypotheses..."
        vpHypo = self.getVPHypVia2Lines()

        print "Get sphere grid..."
        sphereGrid = self.getSphereGrids()

        print "Test vp hypotheses..."
        vps = self.getBestVpsHyp(sphereGrid, vpHypo)

        if np.linalg.det(vps) < 0:
            vps = [vps[0], vps[2], vps[1]]

        print "Get final line clusters..."
        thAngle = 6.0 / 180.0 * np.pi
        clusters = self.lines2Vps(thAngle, vps)

        clustersedNum = sum([len(clusters[i]) for i in xrange(len(clusters))])

        print "total: %d, clustered: %d \n" % (len(self.lines), clustersedNum)
        print "X: %d, Y: %d, Z:%d \n" % (len(clusters[0]), len(clusters[1]), len(clusters[2]))

        return vps, clusters

    def getVPHypVia2Lines(self):

        num = len(self.lines) # number of line segments

        p   = 1.0 / 3.0 * (1.0 - self.noiseRatio)**2

        confEfficience = 0.9999

        it = np.int(np.log( 1.0 - confEfficience ) / np.log( 1.0 - p ))

        numVp2 = 360

        stepVp2 = 2.0 * np.pi / numVp2

        # get the parameters of each line
        for i in xrange(num):
            lineInfo = []
            p1 = np.array([self.lines[i][0], self.lines[i][1], 1.0])
            p2 = np.array([self.lines[i][2], self.lines[i][3], 1.0])

            # line coefficient via p1 and p2
            lineInfo.append(np.cross(p1, p2))

            dx = self.lines[i][0] - self.lines[i][2]
            dy = self.lines[i][1] - self.lines[i][3]

            # line length
            lineInfo.append(np.sqrt(dx**2+dy**2))

            # line orientation
            orientation = np.arctan2(dy, dx)

            if orientation < 0:
                orientation = orientation + np.pi

            lineInfo.append(orientation)

            self.lineInfos.append(lineInfo)

        del lineInfo

        # get vp hypothesis for each iteration
        vpHypo = [[[] for i in xrange(3)] for i in xrange(it*numVp2)]

        count = 0

        for i in xrange(it):

            idx1 = np.random.randint(num)
            idx2 = np.random.randint(num)

            # get the vp1
            vp1_Img = np.cross(self.lineInfos[idx1][0], self.lineInfos[idx2][0])

            while vp1_Img[2] == 0.0:
                idx2 = np.random.randint(num)
                vp1_Img = np.cross(self.lineInfos[idx1][0], self.lineInfos[idx2][0])

            # first vanishing point (transferred to camera coordinate system)
            vp1 = np.array([ vp1_Img[0] / vp1_Img[2] - self.pp[0], vp1_Img[1] / vp1_Img[2] -self.pp[1], self.f])

            if vp1[2] == 0.0:
                vp1[2] = 0.0011

            # normalise vp1
            vp1 = vp1 / np.linalg.norm(vp1)

            # initiate vp2 and vp3
            vp2 = np.zeros(3)
            vp3 = np.zeros(3)

            for j in xrange(numVp2):
                # vp2
                lamb = j * stepVp2
                k1 = vp1[0] * np.sin(lamb) + vp1[1] * np.cos(lamb)
                k2 = vp1[2]
                phi = np.arctan( - k2 / k1 )

                Z = np.cos(phi)
                X = np.sin(phi) * np.sin(lamb)
                Y = np.sin(phi) * np.cos(lamb)

                vp2[0] = X
                vp2[1] = Y
                vp2[2] = Z

                if vp2[2] == 0.0:
                    vp2[2] = 0.0011
                # normalise vp2
                vp2 = vp2 / np.linalg.norm(vp2)

                if vp2[2] < 0.:   # should be refined
                    vp2 = -1. * vp2

                # vp3
                vp3 = np.cross(vp1, vp2)
                if vp3[2] == 0.0:
                    vp3[2] = 0.0011

                # normalise vp3
                vp3 = vp3 / np.linalg.norm(vp3)
                if vp3[2] < 0.:
                    vp3 = -1. * vp3

                vpHypo[count][0] = [vp1[0], vp1[1], vp1[2]]
                vpHypo[count][1] = [vp2[0], vp2[1], vp2[2]]
                vpHypo[count][2] = [vp3[0], vp3[1], vp3[2]]

                count = count + 1

        return vpHypo

    def getSphereGrids(self):

        # build sphere grid with 1 degree accuracy
        angelAccuracy = 1.0 / 180.0 * np.pi
        angleSpanLA = np.pi / 2.0
        angleSpanLO = np.pi * 2.0
        gridLA = np.int(angleSpanLA / angelAccuracy)
        gridLO = np.int(angleSpanLO / angelAccuracy)
        # initiate sphereGrid
        sphereGrid = np.zeros([gridLA, gridLO])

        # put intersection points into the grid
        angelTolerance = 60.0 / 180.0 * np.pi

        for i in xrange(len(self.lines)-1):
            for j in xrange(i+1, len(self.lines)):

                ptIntersect = np.cross(self.lineInfos[i][0], self.lineInfos[j][0])

                if ptIntersect[2] == 0.:
                    continue

                angleDev = np.abs(self.lineInfos[i][2] - self.lineInfos[j][2])
                angleDev = min(np.pi-angleDev, angleDev)
                if angleDev > angelTolerance:
                    continue

                X = ptIntersect[0] / ptIntersect[2] - self.pp[0]
                Y = ptIntersect[1] / ptIntersect[2] - self.pp[1]
                Z = self.f

                latitude  = np.arccos( Z / np.sqrt( X**2 + Y**2 + Z**2 ) )
                longitude = np.arctan2(X, Y) + np.pi

                LA = np.int(latitude / angelAccuracy) # measure in degree

                if LA >= gridLA: # check whether out-bound
                    LA = gridLA - 1

                LO = np.int(longitude / angelAccuracy) # measure in degree
                if LO >= gridLO:
                    LO = gridLO - 1

                sphereGrid[LA][LO] += np.sqrt(self.lineInfos[i][1] * self.lineInfos[j][1]) * (
                            np.sin(2.0 * angleDev) + 0.2) # 0.2 is much robuster

        # Gaussian filter
        halfSize = 1
        winSize = halfSize * 2 + 1
        neighNum = winSize * winSize

        sphereGridNew = np.zeros([gridLA, gridLO])

        for i in xrange(halfSize, gridLA-halfSize):
            for j in xrange(halfSize, gridLO-halfSize):

                neighborTotal = 0.0

                for m in range(0, winSize):
                    for n in range(0, winSize):
                        neighborTotal += sphereGrid[i-halfSize+m][j-halfSize+n]

                sphereGridNew[i][j] = sphereGrid[i][j] + neighborTotal / neighNum

        return sphereGridNew

    def getBestVpsHyp(self, sphereGrid, vpHypo):

        num = len(vpHypo)
        oneDegree = 1.0 / 180.0 * np.pi
        lineLength = np.zeros(num)

        for i in xrange(num):
            for j in xrange(3):

                if vpHypo[i][j][2] == 0.0:
                    continue
                if (vpHypo[i][j][2] > 1.0) or (vpHypo[i][j][2] < -1.0):
                    print "Some vanishing point hypothese is wrongly calculated."

                latitude  = np.arccos(vpHypo[i][j][2])
                longitude = np.arctan2(vpHypo[i][j][0], vpHypo[i][j][1]) + np.pi

                gridLA = np.int(latitude / oneDegree)

                if gridLA == 90:
                    gridLA == 89

                gridLO = np.int(longitude / oneDegree)

                if gridLO == 360:
                    gridLO = 359

                lineLength[i] += sphereGrid[gridLA][gridLO]

        # get the best hypotheses
        bestIdx = np.argmax(lineLength)

        return vpHypo[bestIdx]

    def lines2Vps(self, thAngle, vps ):
        clusters = [[] for i in xrange(3)]

        vp2D = [[] for i in xrange(3)]
        for i in xrange(3):
            vp2D[i] = np.array([vps[i][0] * self.f / vps[i][2] + self.pp[0], vps[i][1] * self.f / vps[i][2] + self.pp[1]])

        for i in xrange(len(self.lines)):

            x1 = self.lines[i][0]
            y1 = self.lines[i][1]
            x2 = self.lines[i][2]
            y2 = self.lines[i][3]

            pt1 = np.array([x1, y1])
            pt2 = np.array([x2, y2])
            ptm = (pt1 + pt2) / 2.

            vc = (pt1 - pt2) / (np.linalg.norm(pt1 - pt2))

            minAngle = 1000.
            bestIdx = None

            for j in xrange(3):
                vp2d_c = vp2D[j] - ptm
                vp2d_c = vp2d_c / np.linalg.norm(vp2d_c)

                dotValue = np.dot(vp2d_c, vc)

                if dotValue > 1.0:
                    dotValue = 1.0
                if dotValue < -1.0:
                    dotValue = -1.0

                angle = np.arccos(dotValue)
                angle = min(np.pi-angle, angle)


                if angle < minAngle:
                    minAngle = angle
                    bestIdx = j

            if minAngle<thAngle:
                clusters[bestIdx].append(i)
        return clusters






















