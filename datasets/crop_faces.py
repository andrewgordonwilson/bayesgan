#!/usr/bin/env python

import os
import numpy as np
import argparse

from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import cv2
import sys


def face_detect(image):
    cascPath = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces if len(faces) == 1 else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to crop images in celebA')
    parser.add_argument('--data_path',
            		type=str, 
	    		required=True, 
			help='location of the data')
    args = parser.parse_args()
    celeb_path = os.join(args.data_path, 'celebA/img_align_celeba')
    print(celeb_path)

    num_images = 202599

    for img_idx in range(num_images):
        X = imread(os.path.join(celeb_path, ("%06d.jpg" % img_idx)))
        faces = face_detect(np.copy(X))
        if faces is not None:
            x, y, w, h = faces[0]
            X_face = X[y:y+h, x:x+h, :]
            X_face = imresize(X_face, size=(32, 32), interp="bicubic")
            imsave(os.path.join(celeb_path, ("%06d_cropped.jpg" % img_idx)), X_face)
            print "Wrote %s" % os.path.join(celeb_path, "%06d_cropped.jpg" % img_idx)
            
            
