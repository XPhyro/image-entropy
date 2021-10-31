# Bugs

# Features
- Evaluate the entropy on each colour channel, then join the results.
- Implement different methods.
  - `1d-shannon`: Standard Shannon entropy just for comparison.
  - `2d-regional-delentropy`: Same as `2d-delentropy`, but using kernels.
- In `1d-kapur`, execute the threshold multiple times and get an iterative entropy image. Then, plot it instead of the threshold image.

# Refactor / Rework / Optimisation
- Optimise `2d-regional-shannon`. Ideally, it would be consistent of only library code instead of Python loops.

# Other
