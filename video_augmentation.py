############################################################################
########################### REQUIRED PACKAGES# #############################
############################################################################

import cv2
import numpy as np
import time
import re
import os

from random import random
from random import randint


############################################################################
############################## AUGMENTATE ##################################
############################################################################

def augmentate(input_dir, output_dir,
               input_format, output_format, show_video=True,
               save_video=False, slow=False, show_size=False,
               seconds_before_action=0, transformations=["aff"]):
    # This function takes all the files in input_dir and, after applying the transformations, saves them in output_dir.

    ######################################################
    # WORKING DIRECTORY
    files = os.listdir(input_dir)
    ######################################################
    exp = re.compile('.*\.' + input_format + '$')
    files_name = [s for s in files if exp.match(s)]
    print("Working with the following files in '",
          input_dir, "': ", files_name, sep="")

    if show_video:
        print("Press 'q' to stop playing\n")

    for input_data in files_name:
        if seconds_before_action > 0:
            print("Reading ", input_data, " from second ",
                  seconds_before_action, "...", sep='')
        else:
            print("Reading", input_data, "...")

        ######################################################
        # VARIABLES
        ######################################################
        cap = cv2.VideoCapture(input_dir + input_data)
        frame_time = 0
        once = True  # TODO: REMAKE CODE TO AVOID THIS

        # Additional functions
        def er(n, sigma=3):
            return randint(n-sigma, n+sigma)

        # FPS
        if show_video or save_video:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fps_float = cap.get(cv2.CAP_PROP_FPS)
            spf = 1 / fps

        # First frame
        ret, frame = cap.read()

        if ret:
            # AFFINE TRANSFORMATION
            ######################################################
            if "aff" in transformations:
                points1 = np.float32([
                    [er(50), er(50)],
                    [er(100), er(50)],
                    [er(50), er(200)]])
                points2 = np.float32([[er(a, 10), er(b, 10)]
                                     for [a, b] in points1])
                M = cv2.getAffineTransform(
                    points1, points2)  # Transformation matrix
                h, w, c = frame.shape

        # For visualization
            if show_video and show_size:
                # Before
                height_before, width_before, c = frame.shape
                height_before = str(height_before)
                width_before = str(width_before)

            ######################################################
            # SAVING THE VIDEO
            ######################################################
            if save_video:
                output_format = "." + output_format
                output_data = output_dir + input_data  # Saved with the same name

                if(output_format == ".mp4"):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif(output_format == ".avi"):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # Exit, format, fps, resolution
                out = cv2.VideoWriter(output_data, fourcc, fps_float, (h, w))

        ######################################################
        # MAIN LOOP
        ######################################################
        while cap.isOpened():

            if not ret:
                break

            if (frame_time > seconds_before_action):
                if slow:
                    time.sleep(spf)  # For visualization

                if show_video:
                    frame_before = frame  # Saved for visualization

                ######################################################
                # TRANSFORMATIONS
                ######################################################

                # AFFINNE TRANSFORMATION
                ######################################################
                frame = cv2.warpAffine(frame, M, (w, h))
                if once:
                    # After
                    height, width, c = frame.shape
                    height = str(height)
                    width = str(width)
                    once = False  # Never again

                ## TODO: UPSAMPLING & DOWNSAMPLING
                ######################################################
                # Xopre: I believe that this should go in the save/show the frame section

                ## TODO: DARKEN & LIGHTEN
                ######################################################

                # TODO: RANDOM SALT/PEPPER NOISE
                ######################################################

                ######################################################
                # SAVE THE FRAME
                ######################################################
                if save_video:
                    out.write(frame)

                ######################################################
                # SHOW BOTH VIDEOS
                ######################################################
                if show_video:
                    both = np.concatenate((frame_before, frame), axis=1)
                    cv2.putText(img=both, text="frame_time=" + str(round(frame_time, 2)), org=(
                        20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 350, 0), thickness=2)
                    # TODO: Adapt frame_time

                    if show_size:
                        cv2.imshow(input_data + ' Original ' + height_before + 'x' +
                                   width_before + ' vs Procesada ' + height + 'x' + width, both)
                    else:
                        cv2.imshow(input_data + ' Original vs Procesada', both)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # To stop visualization
                break

        ######################################################
        # UPDATE VARIABLES
        ######################################################
            # if (frame_time > seconds_before_action):
            frame_time += spf  # Time flies
            ret, frame = cap.read()  # Next frame
        # while cap.isOpened():
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
