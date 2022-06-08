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
############################## AUGMENTATE ##################################
############################################################################
def augment(input_dir, output_dir, 
input_format, output_format, show_video = True, 
save_video = False, slow = False, show_size = False, 
seconds_before_action = -1, transformations = ["aff"]):
    # This function takes all the files in input_dir and, after applying the transformations, saves them in output_dir.

    ######################################################
    ## WORKING DIRECTORY
    files = os.listdir(input_dir)
    ######################################################
    exp = re.compile('.*\.' + input_format + '$')
    files_name = [s for s in files if exp.match(s)]
    print("Working with the following files in '", input_dir,"': ", files_name, sep = "")

    if show_video:
        print("Press 'q' to stop playing\n")

    salt_or_pepper = "bsalt" in transformations or "bpepper" in transformations or "asalt" in transformations or "apepper" in transformations

    ######################################################
    #### VARIABLES (I)
    ######################################################

    cap_example = cv2.VideoCapture(input_dir + files_name[0]) # The first one
    ret, frame = cap_example.read()
    if ret:
        if salt_or_pepper:
            rows, cols, channels = frame.shape
            n_mats = 20
            rand_mats = [np.random.rand(rows, cols) for i in range(n_mats)] # n_mats = 100 composiciones matriciales diferentes
            cap_example.release()

        if "blur" in transformations:
            kernel = np.ones((5,5),np.float32)/25

    else:
        print("Problem reading the first file")
        exit()




    ######################################################
    #### FOR EACH FILE
    ######################################################
    for input_data in files_name:
        if seconds_before_action > 0:
            print("Reading ", input_data, " from second ", seconds_before_action, "...", sep='')
        else: 
            print("Reading", input_data, "...")

        ######################################################
        #### VARIABLES (II)
        ######################################################
        cap = cv2.VideoCapture(input_dir + input_data)
        n_frame = 0
        frame_time = 0
        once = True
  
        ## FPS
        if show_video or save_video:
            fps_float = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps_float)
            spf = 1 / fps

        ### TODO: Remake this to avoid checking the first frame, using the following:
            # frame_width = int(cap.get(3))
            # frame_height = int(cap.get(4))
        ## First frame (out of the main loop)
        ret,frame = cap.read() 
        new_frame = frame # For transformations
        frame_before = frame # & visualization

        if ret:
            ######################################################
            ### SAVE AND SHOW
            ######################################################
            if show_size or save_video:
                # Before
                height_before, width_before, channels_before = frame.shape
                height_before_str = str(height_before)
                width_before_str = str(width_before)
                # TODO: DELETE THIS
                if once:
                    print("height_before = ", height_before, "width_before = ", width_before)

            ## AFFINE TRANSFORMATION
            ######################################################
            if "aff" in transformations:
                points1 = np.float32([
                                    [er(50), er(50)],
                                    [er(100), er(50)],
                                    [er(50), er(200)]])
                points2 = np.float32([[er(a, 10), er(b, 10)] for [a,b] in points1])
                M = cv2.getAffineTransform(points1,points2) # Transformation matrix
                h,w,c = frame_before.shape
                # TODO: DELETE THIS
                if once:
                    print("h = ", h, "w = ", w)

            ######################################################
            ### SAVING THE VIDEO
            ######################################################
            if save_video:
                output_format = "." + output_format # TODO: Use this to generate custom real format
                output_data = output_dir + input_data # Saved with the same name AND THE SAME FORMAT
                # TODO: DELETE THIS
                if once:
                    print("output_data = ", output_data)

                if(output_format == ".mp4"): # Only this one works. Read the TODO above.
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif(output_format == ".avi"):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_data, fourcc, fps_float, (width_before, height_before)) # Exit, format, fps, resolution
        
        ######################################################
        ### MAIN LOOP
        ######################################################
        while cap.isOpened():
            
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
                    new_frame = cv2.warpAffine(new_frame,M,(w,h))
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
                    if once:
                    # TODO: DELETE THIS
                        print("once & save_video")
                        print("frame_before.shape = ", frame_before.shape)
                        print("new_frame.shape = ", new_frame.shape)

                ######################################################
                ### SHOW BOTH VIDEOS
                ######################################################
                if show_video:
                    both = np.concatenate((frame_before, new_frame), axis=1)
                    cv2.putText(img=both, text="frame_time=" + str(round(frame_time, 2)), org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 350, 0), thickness=2)
                    # TODO: Adapt frame_time

                    if show_size:
                        if once:
                            # After all the transformations
                            height, width, c = new_frame.shape
                            height_str = str(height)
                            width_str = str(width)
                        cv2.imshow(input_data + ' Original ' + height_before_str + 'x' + width_before_str + ' vs Procesada ' + height_str + 'x' + width_str, both)
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
            frame_before = frame # Saved for visualization 
            new_frame = frame # Saved for visualization 
            n_frame += 1
        # while cap.isOpened():
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()