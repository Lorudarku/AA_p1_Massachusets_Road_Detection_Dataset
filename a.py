import numpy as np
import math
import cv2
import os



def main():

    folder = './it/'

    for filename in os.listdir(folder):

        name = filename[:-5]
        print(name)

        inImage = cv2.imread('./it/' + filename)
        outImage = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY) 

        (x, y) = np.shape(outImage)
            
        #Subdivido el problema en 9 regiones , las imagenes de prueba son de 1500x1500.
        numsplit = 2
        subx = x // numsplit
        suby = y // numsplit

        for i in range(0, numsplit):
            for j in range(0, numsplit):
                roi = inImage[i*suby:i*suby + suby ,j*subx:j*subx + subx]


                #######################################################
                cv2.imwrite('./it_c/' + name + str(i) + '_' + str(j) + '.tiff', roi)
                #######################################################

main()