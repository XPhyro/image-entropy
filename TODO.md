# Bugs
- `1d-shannon` does not work correctly.

# Features
- Evaluate the entropy on each colour channel, then join the results.
- In `1d-kapur`, execute the threshold multiple times and get an iterative entropy image. Then, plot it instead of the threshold image.
- Implement different methods.
  - `1d-weighted-kapur`: Same as `1d-kapur`, but weighted by thresholded pixel count.
  - `2d-kapur`: Same as `1d-kapur`, but in two dimensions.
  - `2d-weighted-kapur`: Same as `2d-weighted-kapur`, but in two dimensions.
  - `2d-regional-delentropy`: Same as `2d-delentropy`, but using kernels.
- Add a rotated gradient test image.

# Refactor / Rework / Optimisation
- Optimise `2d-regional-shannon`. Ideally, it would be consistent of only library code instead of Python loops.

# Other
- Write a function that *regional*ises methods.
- Add a `CITATION.cff` file.
