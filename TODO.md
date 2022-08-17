# All
- Add a `CITATION.cff` file.
- Add sub-readmes.
- Set up Docker configs for each subproject.
- Add unit tests via GitHub Workflow & Codeberg CI.


# 1 - Assessment

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
- Add dynamic hash/entropy table support for methods. If the hash of the image
  exists in a dynamically managed table in some file, use the result of that
  instead of computing.
  - Load the table to memory. Don't read the file on demand.
  - Make sure there is some locking mechanism between different instances to
    ensure the table will not be corrupted.

## Refactor / Rework / Optimisation
- Optimise `2d-regional-shannon`. Ideally, it would be consistent of only
  library code instead of Python loops.

## Other
- Write a function that *regional*ises methods.


# 2 - Segmentation

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


# 7 - Extraction

## Features
- Implement dynamically adjusted multi-stacking.
  - [X] Create a subprocess for each stack.
  - [ ] Pipe the source video to each stack, and prioritise offset frames in each
    stack.
  - [ ] If entropy is too low with offset prioritisation, try shuffling offsets.
  - [ ] Try writing the assessment part in C/C++, and calling the command from
    Python, consistent with the aforementioned subprocessed stack management.
- Incorporate `XPhyro/scripts/fmapc` and accept hex pi files (as a separate
  option or automatically detected?).
- Assess `/dev/urandom` reference multiple times and average instead of
  one-shot.
- Accept `ffmpeg` options. Options that change essential parameters that are
  obtained via `ffprobe` such as `-filter:v "crop=w:h:x:y"` should be paid
  attention to.
  - Maybe obtain a sample frame from `ffmpeg` and set parameters from that
    frame?
- Test entropy bits before printing:
  - [ ] Normalised mean XOR should be close to 1.
