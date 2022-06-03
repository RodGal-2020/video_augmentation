# video_augmentation
 Recopilation of tools for data augmentation with videos

## How to use this repository
Just download the `video_augmentation.py` file, copy it in your working directory and use:
```
import video_augmentation as va
```
Now everything is set to use all the functionalities of the repository!

Try, for example, the following code:
```
import video_augmentation as va
seed(1974)

## Folders
input_dir = "data/"
# input_file = "data/example.mp4" # TODO
input_format = "mp4"

output_dir = "data/augmented/"
output_format = "mp4" # Also accepts "avi"

## Booleans
save_video = False # Save the video in output_dir?
show_size = True # Show the size of the video in the title?
show_video = True # Show the video at all?
slow = True # Play the video at real rate or at opencv speed?

## Other parameters
seconds_before_action = 0 # Seconds before watching the video

### Execution
va.augmentate(input_dir, output_dir, input_format, output_format, show_video, save_video, slow, show_size, seconds_before_action, ["aff"])
```

## What can you do with this repository?
![Work in progress](data/work_in_progress.png)

## Used data
* example_jpg taken from [Pixabay](https://pixabay.com/es/photos/globo-farolillos-chinos-linterna-3206530/)
* example.mp4 taken from [Pixabay](https://pixabay.com/es/videos/truco-motos-sincr%C3%B3nico-extremo-1083/)
* work_in_progress.png taken from [Pixabay](https://pixabay.com/es/vectors/trabajo-en-progreso-firmar-actividad-24027/)