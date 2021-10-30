# Bugs

# Features
- Evaluate the entropy on each colour channel, then join the results.
- Implement different methods.
  - `1d-shannon`: Standard Shannon entropy just for comparison.
  - `2d-regional-delentropy`: Same as `2d-delentropy`, but using kernels.

# Refactor / Rework / Optimisation
- Optimise `pseudo-spatial`. Ideally, it would be consistent of only library code instead of Python loops.
- Rework plotting not to duplicate code.
- Instead of reading the image in the method, read in main not to duplicate code.

# Other
- Rename `pseudo-spatial` to `2d-regional-shannon`.
