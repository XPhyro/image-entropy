# All
- Add a `CITATION.cff` file.
- Add sub-readmes.
- Set up Docker configs for each subproject.


# CNN Test
None.


# Output
None.


# Development

## Bugs
- `*/development/*` seems to be broken. It enters a memory allocation loop
  somewhere and dies after running out of memory.

## Machine Learning
- See if we need a background category

## Features
- Automatically determine sigma.
- Automatically determine mu.
- Automatically determine kernel size.
- If FILE is a directory, read all files (non-recursively) in it. Have an option
  to make this recursive.
- Output two other ROI images using the segmented object instances:
  - Compute the entropy globally and split to instances.
  - Compute the entropy in each object.
- Before entropy and other computations, segment all images, then distribute the
  segmentations to the deployed CPU processes.
- Make multi-device segmentation simultaneous instead of consecutive.
- Isolate TensorFlow configuration and segmentation worker(s) from the main
  process to force TensorFlow to free device memory. Currently, the TPU/GPU/CPU
  memory stays allocated to the main process even after the segmentation.


# Research

## Features
- Evaluate the entropy on each colour channel, then join the results.
- In `1d-kapur`, execute the threshold multiple times and get an iterative
  entropy image. Then, plot it instead of the threshold image.
- Implement different methods.
  - `1d-weighted-kapur`: Same as `1d-kapur`, but weighted by thresholded pixel
    count.
  - `2d-kapur`: Same as `1d-kapur`, but in two dimensions.
  - `2d-weighted-kapur`: Same as `2d-weighted-kapur`, but in two dimensions.
  - `2d-regional-delentropy`: Same as `2d-delentropy`, but using kernels.
- Add a rotated gradient test image.
- Add command-line options for `mu` and `sigma` in `delentropy2dvc`.

## Refactor / Rework / Optimisation
- Optimise `2d-regional-shannon`. Ideally, it would be consistent of only
  library code instead of Python loops.

## Other
- Write a function that *regional*ises methods.
