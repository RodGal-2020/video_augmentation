############################################################################
########################### REQUIRED PACKAGES ##############################
############################################################################

import cv2
import numpy as np
import time
import re
import os

from random import random
from random import randint


############################################################################
############################ AUX FUNCTIONS #################################
############################################################################
def show_img(img, text = "Imagen"):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def er(n, sigma = 3):
    return randint(n-sigma, n+sigma)

def noise(frame, shape, n_mats, rand_mats, prob = 0.15, verbose = False, n_frame = 0, spice = "pepper"):
    rows, cols, channels = shape
    r_rows = range(rows)
    r_cols = range(cols)
    r_channels = range(channels)
    
    if spice == "pepper":
        for i in r_rows:
            for j in r_cols:
                if rand_mats[n_frame % n_mats][i,j] < prob:
                    for k in r_channels:
                        frame[i][j][k] = 0
    elif spice == "salt":
        for i in r_rows:
            for j in r_cols:
                if rand_mats[n_frame % n_mats][i,j] < prob:
                    for k in r_channels:
                        frame[i][j][k] = 255
                        
    return frame


############################################################################
################################ AUGMENT ###################################
############################################################################
def augment(input_dir, output_dir, 
input_format, output_format, show_video = True, 
save_video = False, slow = False, show_size = False, 
seconds_before_action = -1, transformations = ["aff"], n_mats = 20,
debug_mode = False):
    # This function takes all the files in input_dir and, after applying the transformations, saves them in output_dir.

    ######################################################
    ## WORKING DIRECTORY
    ######################################################
    files = os.listdir(input_dir)

    exp = re.compile('.*\.' + input_format + '$')
    files_name = [s for s in files if exp.match(s)]
    print("Working with the following files in '", input_dir,"': ", files_name, sep = "")

    if show_video:
        print("Press 'q' to stop playing\n")


    ######################################################
    ### TRANSFORMATIONS & VARIABLES
    ######################################################
    cap_example = cv2.VideoCapture(input_dir + files_name[0]) # The first one
    ret, frame = cap_example.read()
    if ret:
        ## COMMON VARIABLES
        ######################################################
        # We assume that all videos have the same dimensions
        video_width = int(cap_example.get(3))
        video_height = int(cap_example.get(4))

        ## SALT & PEPPER
        ######################################################
        salt_or_pepper = "bsalt" in transformations or "bpepper" in transformations or "asalt" in transformations or "apepper" in transformations
        if salt_or_pepper:
            rand_mats = [np.random.rand(video_height, video_width) for i in range(n_mats)]
            cap_example.release()

        ## BLUR TRANSFORMATION
        ######################################################
        if "blur" in transformations:
            kernel = np.ones((5,5), np.float32) / 25
            
    else:
        print("Problem reading the first file")
        exit()




    ######################################################
    #### FOR EACH FILE
    ######################################################
    for input_data in files_name:
        if seconds_before_action > 0:
            print("\nReading ", input_data, " from second ", seconds_before_action, "...", sep='')
        else: 
            print("\nReading", input_data, "...")

        ######################################################
        #### VARIABLES
        ######################################################
        ## VIDEO INFO
        cap = cv2.VideoCapture(input_dir + input_data)
        # We assume that all videos have the same dimensions
        ## AUX VARIABLES
        n_frame = 0 # Frame number in this video. Required for the noise() function, in order to choose a rand_mat
        frame_time = 0 # Time already shown
        once = True # To show certain lines, but only once per file
  
        ## FPS
        # Supposing that FPS are the same always can lead to error (look at "SAVING THE VIDEO")
        if show_video or save_video:
            fps_float = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps_float)
            spf = 1 / fps


        ## SAVING THE VIDEO
        if save_video:
            output_format = "." + output_format # TODO: Use this to generate custom real format
            output_data = output_dir + input_data # Saved with the same name AND THE SAME FORMAT
            if once & debug_mode:
                print("output_data = ", output_data)

            if(output_format == ".mp4"): # Only this one works. Read the TODO above.
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif(output_format == ".avi"):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_data, fourcc, fps_float, (video_width, video_height)) # Exit, format, fps, resolution

        ## AFFINE TRANSFORMATION
        ######################################################
        # It should be different for each video
        if "aff" in transformations:
            points1 = np.float32([
                                [er(50), er(50)],
                                [er(100), er(50)],
                                [er(50), er(200)]])
            points2 = np.float32([[er(a, 10), er(b, 10)] for [a,b] in points1])
            M = cv2.getAffineTransform(points1, points2) # Transformation matrix


        ######################################################
        ### MAIN LOOP
        ######################################################
        cap = cv2.VideoCapture(input_dir + input_data) # We open it again to avoid conflicts with the previous step
        ret, frame = cap.read()

        while cap.isOpened():
            new_frame = frame
            frame_before = frame

            if not ret:
                break

            if (frame_time > seconds_before_action):
                if slow:
                    time.sleep(spf) # For visualization

                ######################################################
                ### TRANSFORMATIONS
                ######################################################

                ### RANDOM SALT/PEPPER NOISE BEFORE
                ######################################################                
                if "bpepper" in transformations:
                    new_frame = noise(frame = new_frame, shape = new_frame.shape, n_mats = n_mats, rand_mats = rand_mats, verbose = True, n_frame = n_frame, spice = "pepper")
                    if once: 
                        print("Applying before_pepper")
                if "bsalt" in transformations:
                    new_frame = noise(frame = new_frame, shape = new_frame.shape, n_mats = n_mats, rand_mats = rand_mats, verbose = True, n_frame = n_frame, spice = "salt")
                    if once: 
                        print("Applying before_salt")

                ### AFFINE TRANSFORMATION
                ######################################################
                if "aff" in transformations:
                    new_frame = cv2.warpAffine(new_frame,M,(video_width,video_height))
                    if once: 
                        print("Applying affine_transformation")

                ## TODO: UPSAMPLING & DOWNSAMPLING
                ######################################################
                # Xopre: I believe that this should go in the save/show the frame section


                ## TODO: DARKEN & LIGHTEN
                ######################################################

                ## BLUR & MEDIAN BLUR
                ######################################################
                if "blur" in transformations:
                    new_frame = cv2.filter2D(new_frame, -1, kernel)
                    if once: 
                        print("Applying blur")
                if "mblur" in transformations:
                    new_frame = cv2.medianBlur(new_frame, 5)
                    if once: 
                        print("Applying median_blur")

                ## RANDOM SALT/PEPPER NOISE AFTER
                ######################################################
                if "apepper" in transformations:
                    new_frame = noise(frame = new_frame, shape = new_frame.shape, n_mats = n_mats, rand_mats = rand_mats, verbose = True, n_frame = n_frame, spice = "pepper")
                    if once: 
                        print("Applying after_pepper")
                if "asalt" in transformations:
                    new_frame = noise(frame = new_frame, shape = new_frame.shape, n_mats = n_mats, rand_mats = rand_mats, verbose = True, n_frame = n_frame, spice = "salt")
                    if once: 
                        print("Applying after_salt")

                ######################################################
                ### SAVE THE FRAME
                ######################################################
                if save_video:
                    out.write(new_frame)
                    if once & debug_mode:
                        print("once & save_video")
                        print("frame_before.shape = ", frame_before.shape)
                        print("new_frame.shape = ", new_frame.shape)

                ######################################################
                ### SHOW BOTH VIDEOS
                ######################################################
                if show_video:
                    both = np.concatenate((frame_before, new_frame), axis=1)
                    cv2.putText(img=both, text="frame_time=" + str(round(frame_time, 2)), org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 350, 0), thickness=2)
                    # FIXME: Adapt frame_time

                    if show_size:
                        if once:
                            if debug_mode:
                                print("show size once")
                        cv2.imshow(input_data + ' Original vs Procesada ' + str(video_width) + 'x' + str(video_height), both)
                    else:
                        cv2.imshow(input_data + ' Original vs Procesada', both)
       
                ## ONCE
                #################
                once = False # Never again


            if cv2.waitKey(1) & 0xFF == ord('q'): # To stop visualization
                break  

        ######################################################
        ### UPDATE VARIABLES
        ######################################################
        # while cap.isOpened():
            # if (frame_time > seconds_before_action):
            frame_time += spf # Time flies
            ret,frame = cap.read() # Next frame
            n_frame += 1

        # while cap.isOpened():
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()