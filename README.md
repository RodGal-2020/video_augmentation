# video_augmentation
 Recopilation of tools for data augmentation with videos

## Quickstart
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

## Status

- Of the repository:
    - [ ] Complete `README.md`

- Of the main function:
    - [ ] Include a progress bar
    - [ ] Make active use of the GPU to boost performance
    - [ ] Use the `augmentate` code from the SACU-M dataset to create a `multi_augmentate` function

- Included transformations:
    - [x] Affinity
        - [ ] Scaling
        - [ ] Homothety
        - [ ] Translation
        - [ ] Rotation
        - [x] Random w/o control of the randomness
        - [ ] Random w/ control of the randomness
    - [x] Salt & Pepper
        - [x] Pseudorandom
        - [ ] Efficient pseudorandomization
        - [ ] For every size
    - [x] Blur
        - [x] Median-based blur
    - [ ] Upsampling & Downsampling
    - [ ] Darken & Lighten

## Used data
* example.jpg taken from [Pixabay](https://pixabay.com/es/photos/globo-farolillos-chinos-linterna-3206530/)
* example_small.jpg taken from [Pixabay](https://pixabay.com/es/photos/gato-felino-mascota-animal-6960183/)
* example.mp4 taken from [Pixabay](https://pixabay.com/es/videos/truco-motos-sincr%C3%B3nico-extremo-1083/)
* example2.mp4 taken from [Pixabay](https://pixabay.com/es/videos/gallo-pollo-aldea-granja-10685/)
* work_in_progress.png taken from [Pixabay](https://pixabay.com/es/vectors/trabajo-en-progreso-firmar-actividad-24027/)