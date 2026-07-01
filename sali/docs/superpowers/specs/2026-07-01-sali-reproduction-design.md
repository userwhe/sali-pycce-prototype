# SALI Reproduction Design

## Purpose

Build a faithful, PyTorch-based reproduction of "Automatic Detection of Nuclear Spins at Arbitrary Magnetic Fields via Signal-to-Image AI Model" in `/Users/weitao/Code/python/sali`. The reproduction will implement the paper's physics simulation, 1D-to-2D signal-to-image model, training recipe, image post-processing, evaluation metrics, and Colab execution workflow. The default run will be a practical Colab smoke reproduction, while configuration will preserve the paper-scale settings.

The original paper states that code can be requested from the authors, so this project will reproduce the method from the equations and supplemental architecture/training description rather than copying an official implementation.

## Scope

The first implementation will support both high- and low-field scenarios, but it will default to one configurable field mode per run. It will generate synthetic datasets locally or in Colab, train the PyTorch model, evaluate on a held-out test split, and produce the requested plots.

The default run will use a small dataset and reduced epoch count so the pipeline can complete in ordinary Colab. A `paper` config will expose the described paper-scale settings:

- `3_600_000` samples per magnetic-field scenario
- training/validation/test split of `70% / 15% / 15%`
- `N=32` and `N=256` CPMG inputs
- `1000` sampled tau points per signal
- `Bz=0.056 T` for high field and `Bz=0.0056 T` for low field
- batch size `64`
- maximum `250` epochs
- Adam learning rate `0.001`
- learning-rate reduction factor `0.7` after 5 validation-loss plateau epochs
- early stopping patience `20`

## Package Layout

The project will be organized as a Python package plus runnable entrypoints:

- `src/sali/config.py`: dataclasses and presets for practical and paper-scale runs.
- `src/sali/physics.py`: CPMG signal generation from the paper's equations.
- `src/sali/targets.py`: Gaussian heatmap generation on the `204 x 104` output grid.
- `src/sali/data.py`: synthetic sample generation, split creation, normalization, and PyTorch datasets.
- `src/sali/model.py`: PyTorch SALI network.
- `src/sali/train.py`: training loop, checkpointing, learning-rate scheduling, early stopping, and history capture.
- `src/sali/postprocess.py`: morphology, thresholding, connected components, local maxima, and centroid extraction.
- `src/sali/metrics.py`: IoU matching, precision, recall, coupling MAE, and reconstructed-signal MAE.
- `src/sali/plots.py`: notebook/script plotting helpers.
- `scripts/train_colab.py`: Colab CLI entrypoint for smoke and scalable training runs.
- `notebooks/sali_reproduction_colab.ipynb`: Colab-ready notebook for the visual reproduction workflow.
- `tests/`: unit tests for physics, target generation, data handling, model shape, post-processing, and metrics.

## Physics And Data Generation

Each generated sample will choose a random number of nuclei `n` from 1 to 20. For each nucleus, it will sample:

- `A^z in [-100, 100] kHz`
- `A^perp in [2, 102] kHz`

For each sample, the simulator will compute two survival-probability signals `P_x`:

- `N=32`, tau range `[6, 50] us`, 1000 points
- `N=256`, tau range `[10, 40] us`, 1000 points

The implementation will follow the paper's equations for `P_x`, `M_j`, and `cos(phi_j)`, using `gamma_n = 2*pi*10.705 MHz/T`. It will include the described experimental effects:

- exponential decoherence factor with `T2 = 200 us`
- shot noise using `N_m = 1000` measurements by default

The dataset object will retain true nuclei metadata for evaluation and plotting. Training, validation, and testing splits will be generated before normalization. Normalization will use the training-set mean and variance, then apply those statistics to all splits with epsilon `0.001`, matching the supplement.

## Target Heatmaps

The supervised target for each sample will be a single-channel image with PyTorch shape `(1, 204, 104)`. This corresponds to the paper's effective `200 x 100` coupling grid plus a two-pixel border on every side.

The heatmap axes will represent:

- x axis: `A^perp`, from `2` to `102 kHz`
- y axis: `A^z`, from `-100` to `100 kHz`

Each nucleus will be rendered as a Gaussian-like `5 x 5` patch centered at the nearest pixel to its true coupling values, using `exp(-((x-x0)^2 + (y-y0)^2) / 2)`. Pixel values will be clipped to an upper bound of `1.0` so nearby nuclei cannot create values above the paper's specified maximum.

## PyTorch Model

The model will be a faithful PyTorch translation of the described 1D-to-2D CNN.

Inputs:

- `signal_32`: shape `(batch, 1, 1000)`
- `signal_256`: shape `(batch, 1, 1000)`

Each input branch:

- 1D convolution with 16 filters, kernel size 3
- 1D convolution with 32 filters, kernel size 3
- batch normalization
- ReLU
- max pooling with window size 2
- dropout rate `0.2`
- flatten

The flattened branch outputs will be concatenated and passed through the fully connected transition:

- dense bottleneck of size `(102 * 52) / 2`
- batch normalization
- ReLU
- dropout rate `0.2`
- dense layer of size `102 * 52`
- batch normalization
- ReLU
- reshape to `(batch, 1, 102, 52)`

The 2D block:

- 2D convolution with 32 filters and kernel size `(3, 3)`
- batch normalization
- ReLU
- dropout rate `0.2`
- 2D transposed convolution with 16 filters, kernel size `(3, 3)`, stride `2`, and output sizing chosen to produce `204 x 104`
- batch normalization
- ReLU
- output 2D convolution with 1 filter, kernel size `(3, 3)`
- sigmoid activation to constrain pixels to `[0, 1]`

## Training Recipe

The training loop will use:

- MSE loss over all output heatmap pixels
- Adam optimizer with initial learning rate `0.001`
- batch size `64`
- shuffled training batches each epoch
- validation loss after each epoch
- learning-rate reduction after 5 validation-loss plateau epochs, factor `0.7`
- early stopping after 20 validation-loss plateau epochs
- best checkpoint selected by lowest validation loss

The practical default config will use a smaller sample count and fewer epochs to prove the end-to-end pipeline in Colab. That practical run is not expected to reproduce the paper's final metrics, but it will use the same code path as a larger run.

## Image Post-Processing

Predicted heatmaps will be converted to identified C13 spin couplings through:

- erosion and dilation smoothing
- thresholding
- connected-component grouping of adjacent pixels
- area filtering to remove tiny regions
- local-maxima detection inside each retained region
- minimum local-maximum separation of 3 pixels, corresponding to roughly 3 kHz
- centroid extraction for single-nucleus regions
- 5-by-5 boxes around maxima for multi-maximum regions

The routine will return predicted coupling pairs and bounding boxes for evaluation and plotting.

## Evaluation

Evaluation will run on the held-out test split only. For each test sample:

- convert predicted heatmaps to coupling predictions
- construct true `5 x 5` boxes from target nuclei
- match predicted boxes to true boxes using largest IoU
- count a true positive when IoU is greater than 0
- count unmatched predictions as false positives
- count unmatched true nuclei as false negatives
- compute precision and recall by true nucleus count
- compute MAE for `A^z` and `A^perp` over true positives
- reconstruct `N=32` and `N=256` signals from the identified nuclei
- compute reconstructed-signal MAE against the original signals

The implementation will also save aggregate metrics as JSON or CSV in the run directory.

## Notebook Outputs

The Colab notebook will include the requested visual reproduction plots:

- generated spectra for `N=32`
- generated spectra for `N=256`
- true heatmap
- predicted heatmap
- post-processed heatmap
- overlay of original and identified-C13 reconstructed signal for `N=32`
- overlay of original and identified-C13 reconstructed signal for `N=256`
- training and validation loss curves
- precision and recall plots
- MAE plots for `A^z`, `A^perp`, and reconstructed signals

The notebook will call the package code rather than duplicating implementation logic. It will be suitable for Colab execution and compatible with the `colab` CLI workflow.

## Colab CLI Workflow

The first Colab run will use the practical default config. The workflow will:

- create or reuse a named Colab session
- execute `scripts/train_colab.py`
- train on the selected accelerator if one is available
- save the run directory with config, checkpoint, metrics, and plots
- download or expose artifacts for local inspection
- stop the Colab session when work is complete

The implementation will avoid interactive Colab operations. If authentication or accelerator quota fails, the workflow will report the concrete failure and fall back to a smaller or CPU-compatible run where possible.

## Error Handling

The code will validate:

- valid magnetic-field mode
- non-empty train/validation/test splits
- tau ranges and signal length
- finite generated signals
- signal tensor shape `(2, 1000)` before batching
- heatmap tensor shape `(1, 204, 104)`
- model output shape `(batch, 1, 204, 104)`
- finite losses and metrics

Post-processing will handle empty predictions without crashing. Runs will save seed, config, and output paths so results can be reproduced.

## Tests

Tests will focus on deterministic, non-expensive behavior:

- CPMG signal generation returns finite arrays with expected shape and probability-like values.
- target heatmaps place a nucleus inside the expected pixel neighborhood and clip at 1.
- split generation respects train/validation/test sizes.
- normalization uses training-set statistics.
- model forward pass returns `(batch, 1, 204, 104)`.
- post-processing detects simple synthetic Gaussian blobs.
- metric calculation gives expected precision, recall, and MAE on controlled boxes.

Training will also have a smoke test with a very small dataset and one epoch if local dependencies allow it.

## Acceptance Criteria

The reproduction is complete when:

- package modules and tests are implemented in `/Users/weitao/Code/python/sali`
- the PyTorch model forwards successfully with the expected output shape
- the local smoke tests pass or any unavailable dependency is clearly documented
- the Colab CLI practical run completes or reaches a clearly reported external blocker
- the notebook includes all requested plots
- artifacts include trained checkpoint, metrics, and figures from a practical run

