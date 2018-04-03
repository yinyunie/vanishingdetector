import cv2
import numpy as np

class VPDetection:

    # initiate with line segments, principal point, focal length, ratio of outliers

    def __init__(self, lines, pp, f, noiseRatio):

        self.lines = lines
        self.pp = pp
        self.f = f
        self.noiseRatio = noiseRatio
        # para, length, oritentation
        self.lineInfos = []

    def run(self):

        vpHypo = self.getVPHypVia2Lines()
        '''Get vp hypotheses...'''

        vps = []
        clusters = []
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
        vpHypo = [[[] for i in range(3)] for i in range(it*numVp2)]

        count = 0

        for i in range(it):

            idx1 = np.random.randint(num)
            idx2 = np.random.randint(num)

            while idx2 == idx1:
                idx2 = np.random.randint(num)

            # get the vp1
            vp1_Img = np.cross(self.lineInfos[idx1][0], self.lineInfos[idx2][0])

            if vp1_Img[2] == 0.0:
                i = i - 1
                continue

            # first vanishing point (transferred to camera coordinate system)
            vp1 = np.array([ vp1_Img[0] / vp1_Img[2] - self.pp[0], vp1_Img[1] / vp1_Img[2] -self.pp[1], self.f ])

            if vp1[2] == 0.0:
                vp1[2] = 0.0011

            # normalise vp1
            vp1 = vp1 / np.linalg.norm(vp1)

            # initiate vp2 and vp3
            vp2 = np.zeros(3)
            vp3 = np.zeros(3)

            for j in range(numVp2):
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

                if vp2[2] < 0.:
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




