############################################################################
########################### REQUIRED PACKAGES ##############################
############################################################################

import cv2
import numpy as np
import time
import re
import os

# from random import random
from random import randint

############################################################################
########################### PACKAGE INFO ###################################
############################################################################
version = "29/09/2022 - Enigma Epoch"
print("Using the following version of the package:", version)

############################################################################
############################ AUX FUNCTIONS #################################
############################################################################
def set_seed(n):
    '''
    Sets the seed for all random operations
    '''
    # seed(n)
    np.random.seed(n)

def show_img(img, text = "Imagen"):
    '''
    Shows an image until a key is pressed
    '''
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def er(n, sigma = 3):
    '''
    Returns a random int between n-sigma and n+sigma
    '''
    return randint(n-sigma, n+sigma)

## For Salt & Pepper
def sp_noise(image, salt_and_or_pepper, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    
    Adapted from gutierrezps/cv2_noise.py, forked from lucaswiman/cv2_noise.py
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    if "salt" in salt_and_or_pepper and "pepper" in salt_and_or_pepper:
        output[probs < (prob / 2)] = black
        output[probs > 1 - (prob / 2)] = white
    elif "salt" in salt_and_or_pepper:
        output[probs < prob] = white
    else:
        output[probs < prob] = black
    return output

############################################################################
################################ AUGMENT ###################################
############################################################################
def augment(input_dir, output_dir, input_format, output_format, show_video = True, save_video = False, slow = False, show_size = False, seconds_before_action = -1, transformations = [], noise_prob = 0.3, debug_mode = False, log_dir = None, multi = False):
    '''
    This function takes all the files in input_dir and, after applying the transformations, saves them in output_dir.
    '''

    ######################################################
    ## WORKING DIRECTORY
    ######################################################
    files = os.listdir(input_dir)

    if (log_dir is not None) and (not multi):
      save_log = True
    else:
      save_log = False
      
    print("save_log =", save_log)

    if save_log:
      print("Saving log to " + log_dir)
      
      start_time = time.time()
      
      if os.path.exists(log_dir):
        log = open(log_dir, "a")
        log.write("\nNew log for the execution launched the " + time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M:%S") + "\n---------------------------------------------------------------------------")
      else:
        log = open(log_dir, "w")
        log.write("===========================================================================\n")
        log.write("Log for the execution launched the " + time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M:%S") + "\n---------------------------------------------------------------------------")
        
      log.write("\nConfiguration: \n\tinput_dir = %s\n\toutput_dir = %s\n\tinput_format = %s\n\toutput_format = %s\n\tshow_video = %s\n\tsave_video = %s\n\tslow = %s\n\tshow_size = %s\n\tseconds_before_action = %s\n\ttransformations = %s\n\tnoise_prob = %s\n\tdebug_mode = %s\n\tlog_dir = %s" % (input_dir, output_dir, input_format, output_format, show_video, save_video, slow, show_size, seconds_before_action, transformations, noise_prob, debug_mode, log_dir))
      
    exp = re.compile('.*\.' + input_format + '$')
    files_name = [s for s in files if exp.match(s)] # Only the files in the chosen format
    print("Working with the following files in '", input_dir,"': ", files_name, sep = "")

    if show_video:
        print("Press 'q' to stop playing\n")
        
    ######################################################
    ### TRANSFORMATIONS & VARIABLES
    ######################################################
    
    ## BLUR TRANSFORMATION
    ######################################################
    if "blur" in transformations:
        kernel = np.ones((5,5), np.float32) / 25

    ## UPSAMPLING & DOWNSAMPLING TRANSFORMATIONS
    ######################################################
    use_upsampling = False
    use_downsampling = False
    for t in transformations:
        m1 = re.match("usample-(.+)", t)
        m2 = re.match("dsample-(.+)", t)
        if m1 is not None:
            usample_p = float(m1.group(1)) # Probability of upsampling
            use_upsampling = True
        if m2 is not None:
            dsample_p = float(m2.group(1)) # Probability of downsampling
            use_downsampling = True
            
    if use_upsampling and use_downsampling:
        print("Use only usample or dsample, but not both")
        exit()
        
    ######################################################
    #### FOR EACH FILE
    ######################################################
    if save_log:
      log.write("\n---------------------------------------------------------------------------")
      log.write("\nExecution trace:")
      log.write("\n\t- Working with " + str(len(files_name)) + " files")
      log.write("\n\t" + time.strftime("%H:%M:%S") + ": Applying the following transformations: " + ", ".join(transformations) + ".")
    
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
        n_frames = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        frame_time = 0 # Time already shown
        once = True # To show certain lines, but only once per file
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))

        ## FPS
        # Supposing that FPS are the same always can lead to error (look at "SAVING THE VIDEO")
        if show_video or save_video:
            fps_float = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps_float)
            spf = 1 / fps


        ## SAVING THE VIDEO
        if save_video:
            output_format = "." + output_format # TODO: Use this to generate custom real format

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            if transformations == []:
                aug_output_dir = os.path.join(output_dir, "original")
                output_data = os.path.join(aug_output_dir, input_data)
            else:
                aug_output_dir = os.path.join(output_dir, "_".join(transformations))
                output_data = os.path.join(aug_output_dir, input_data)
                
            if not os.path.exists(aug_output_dir):
                  os.mkdir(aug_output_dir)
            
            if once and debug_mode:
                print("aug_output_dir = ", aug_output_dir)
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


        ## UPSAMPLE & DOWNSAMPLE TRANSFORMATIONS
        ######################################################
        # They should be different for each video
        
        if use_downsampling:
            dsample_rand = np.random.rand(n_frames)
            if debug_mode and once: 
                print("Upsample transformation")
            
        if use_upsampling:
            usample_rand = np.random.rand(n_frames)
            if debug_mode and once:
                print("Downsample transformation")

        ######################################################
        ### MAIN LOOP
        ######################################################
        cap = cv2.VideoCapture(input_dir + input_data) # We open it again to avoid conflicts with the previous step
        ret, frame = cap.read()

        while cap.isOpened():

            if not ret:
                break

            new_frame = frame.copy()
            frame_before = frame.copy()

            if (frame_time > seconds_before_action):
                if slow:
                    time.sleep(spf) # For visualization

                ######################################################
                ### TRANSFORMATIONS
                ######################################################

                ### RANDOM SALT/PEPPER NOISE BEFORE
                ######################################################                
                if "bpepper" in transformations:
                    new_frame = sp_noise(new_frame, ["pepper"], noise_prob)
                    if once: 
                        print("Applying before_pepper")
                if "bsalt" in transformations:
                    new_frame = sp_noise(new_frame, ["salt"], noise_prob)
                    if once: 
                        print("Applying before_salt")

                ### AFFINE TRANSFORMATION
                ######################################################
                if "aff" in transformations:
                    new_frame = cv2.warpAffine(new_frame,M,(video_width,video_height))
                    if once: 
                        print("Applying affine_transformation")

                ## UPSAMPLING & DOWNSAMPLING
                ######################################################
                # This goes in the save/show section


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
                    new_frame = sp_noise(new_frame, ["pepper"], noise_prob)
                    if once: 
                        print("Applying after_pepper")
                if "asalt" in transformations:
                    new_frame = sp_noise(new_frame, ["salt"], noise_prob)
                    if once: 
                        print("Applying after_salt")

                ######################################################
                ### SAVE THE FRAME
                ######################################################
                if save_video:
                    if not use_downsampling: 
                        out.write(new_frame)
                    else:
                        print(f"dsample_rand[n_frame] = {dsample_rand[n_frame]}, dsample_rand[n_frame] < dsample_p = {dsample_rand[n_frame] < dsample_p}")
                        if dsample_rand[n_frame] < dsample_p:
                            if once:
                                print("Applying downsampling")
                            pass
                        else:
                            out.write(new_frame)

                    if use_upsampling: 
                        if usample_rand[n_frame] < usample_p: 
                            if once:
                                print("Applying upsampling")
                            out.write(new_frame)
                        
                    if once and debug_mode:
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
    if save_log:
      log.write("\n\t" + time.strftime("%H:%M:%S") + ": All transformations applied.")
      log.write("\n---------------------------------------------------------------------------")
      log.write("\nStats:")
      log.write("\n\t- Total elapsed time: " + str(round(time.time() - start_time, 2)) + " seconds.")
      log.write("\n\t- Mean elapsed time: \n\t\t- per file: " + str(round((time.time() - start_time) / len(files), 2) ) + " seconds.")
      log.write("\n\t\t- per transformation: " + str(round((time.time() - start_time) / len(transformations), 2) ) + " seconds.")
      log.write("\n\t\t- per file & transformation: " + str(round((time.time() - start_time) / len(files) / len(transformations), 2) ) + " seconds.")
      log.write("\n===========================================================================\n")
      log.close()

############################################################################
############################# MULTI-AUGMENT ################################
############################################################################
def multi_augment(input_dir, output_dir, input_format, output_format, show_video = True, save_video = False, slow = False, show_size = False, seconds_before_action = -1, multiple_augmentations = [("train", ["aff"]), ("val", ["apepper", "blur"]), ("test", [])], noise_prob = 0.3, debug_mode = False, log_dir = None):
    '''
    Wrapper function for quick deployment of multiple augmentations throughout the train, validation and test subsets.
    -> Under development.
    '''
    
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
    if log_dir is not None:
      save_log = True
      start_time = time.time()
      
      if os.path.exists(log_dir):
        log = open(log_dir, "a")
        log.write("\nNew multi-augment launched the " + time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M:%S") + "\n---------------------------------------------------------------------------")
      else:
        log = open(log_dir, "w")
        log.write("===========================================================================\n")
        log.write("Log for the multi-augment launched the " + time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M:%S") + "\n---------------------------------------------------------------------------")
        
      log.close()
    
    for subset, transformations in multiple_augmentations:
        new_output_dir = output_dir + "/" + subset + "/"
        if not os.path.exists(new_output_dir):
            os.mkdir(new_output_dir)
            
        augmented_mark = '_'.join(transformations) # None

        print(f"Working with {output_dir} and {augmented_mark} to apply the {transformations} transformations")

        augment(
            input_dir = input_dir, 
            output_dir = new_output_dir, # Modified
            input_format = input_format, 
            output_format = output_format, 
            show_video = show_video, 
            save_video = save_video, 
            slow = slow, 
            show_size = show_size, 
            seconds_before_action = seconds_before_action, 
            transformations = transformations, 
            noise_prob = noise_prob,
            debug_mode = debug_mode,
            log_dir = log_dir,
            multi = True
          ) # Modified
    
    if save_log:
      log = open(log_dir, "a")
      
      all_transformations = [t for (a, t) in multiple_augmentations]
      files = os.listdir(input_dir)
      
      log.write("\n\t- Total elapsed time: " + str(round(time.time() - start_time, 2)) + " seconds.")
      log.write("\n\t- Mean elapsed time per file: " + str(round((time.time() - start_time) / (len(files)), 2)) + " seconds.")
      log.write("\n\t- Mean elapsed time per group of transformations / folder: " + str(round((time.time() - start_time) / len(all_transformations), 2)) + " seconds.")
      
      log.write("\n===========================================================================\n")
      
      log.close()    
    
    print("\033[1;35mmulti_augment execution finished\033[1;0m")
