import math
import numpy as np
from skimage.external.tifffile import imread, imsave
from ArtifactGenerator import Plot

plot = Plot(False)

def pumpingEllipse(k):
    x[k][i ** 2 + 4 * j ** 2 < (25 + 10 * math.sin(k)) ** 2] = 255

def turningEllipse(k):
    x[k][(math.cos(k / 8) * i - math.sin(k / 8) * j) ** 2 + 4 * (math.sin(k / 8) * i + math.cos(k / 8) * j) ** 2 < 25 ** 2] = 255

def largeTurningEllipse(k):
    x[k][(math.cos(k / 4) * i - math.sin(k / 4) * j) ** 2 + 4 * (math.sin(k / 4) * i + math.cos(k / 4) * j) ** 2 < 40 ** 2] = 255

def turningSquare(k):
    x[k][np.abs(math.cos(k / 8) * i - math.sin(k / 8) * j) + np.abs(math.sin(k / 8) * i + math.cos(k / 8) * j) < 25] = 255

def walkingRectangles(k):
    x[k][(np.abs(i)<30) & (np.abs(j+15)<15)] = 255
    x[k][(np.abs(i-20)<10) & (np.abs(j)<20)] = 255
    x[k][(np.abs(i+20)<10) & (np.abs(j)<20+5*math.sin(k/2))] = 255

K = 50
L = 50
i, j = np.meshgrid(range(-L, L+1), range(-L, L+1))
x = np.zeros((K, 2*L+1, 2*L+1), dtype=np.uint8)
for k in range(K):
    # pumpingEllipse(k)
    # largeTurningEllipse(k)
    # turningSquare(k)
    walkingRectangles(k)
    imsave(plot.path + 'Phantom' + str(k + 1) + '.tif', x[k])