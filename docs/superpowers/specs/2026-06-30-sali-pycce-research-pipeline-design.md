# SALI-PyCCE Research Pipeline Design

Date: 2026-06-30

## Goal

Extend the existing SALI-PyCCE prototype into a modular research pipeline that trains an AI model to identify C13 nuclear spin parameters from CPMG data, then tests the model on PyCCE-generated random spin baths and compares it with the local algorithmic decomposition method in `/Users/weitao/Code/python/dqpmodel/cpmg_model`.

The approved strategy is analytic/noisy training plus PyCCE testing. Training data should be fast enough to generate at scale, while PyCCE is used as an independent physics-domain benchmark.

## Current Context

The repository already contains:

- Analytic CPMG simulation in `src/sali_pycce/physics.py`
- Synthetic on-the-fly dataset generation in `src/sali_pycce/data.py`
- Gaussian heatmap labels and decoding in `src/sali_pycce/heatmap.py`
- A compact two-input CNN in `src/sali_pycce/model.py`
- Training and evaluation CLIs in `src/sali_pycce/train.py` and `src/sali_pycce/evaluate.py`
- A starter Colab notebook in `notebooks/train_in_colab.ipynb`
- A PyCCE adapter scaffold in `src/sali_pycce/pycce_backend.py`

Main gaps:

- Research defaults are still prototype-scale.
- The model assumes `32 x 64` output heatmaps.
- Training does not persist metric history for notebook plots.
- PyCCE random-bath testing is not implemented.
- The local `dqpmodel/cpmg_model` decomposition baseline is not wrapped for comparison.

## Architecture

The pipeline should be split into small modules with clear interfaces:

- Analytic/noisy training generator
- Heatmap label generation and decoding
- Shape-flexible SALI model
- Training CLI with persistent metrics
- Visualization/report helpers
- PyCCE random-bath benchmark generator
- Hyperfine tensor projection helper
- Local decomposition baseline adapter
- Shared spin-matching and metric code
- Colab notebook and Colab CLI training workflow

Data flow:

1. Generate analytic CPMG traces with shot noise and NV decoherence.
2. Generate true spin lists and Gaussian heatmap labels.
3. Train the two-input SALI model on `N=32` and `N=256` traces.
4. Decode predicted heatmaps into spin parameter estimates.
5. Generate PyCCE random spin-bath test traces and ground-truth tensor projections.
6. Run the AI model on both PyCCE traces.
7. Run the algorithmic decomposition baseline on the `N=32` PyCCE trace only.
8. Score both methods with the same nearest-neighbor matcher over detectable spins.
9. Display required figures in the notebook and save machine-readable results.

## Training Data And Labels

Training data uses the analytic simulator, not PyCCE.

Default research configuration:

- Magnetic field: `Bz = 525 G`
- CPMG inputs: `N=32` and `N=256`
- Tau range: `0-40 us` for both traces
- Tau samples: `4000` points per trace
- Hyperfine range: `A_z in [-250, 250] kHz`, `A_perp in [2, 250] kHz`
- Heatmap shape: `128 x 256`
- Noise: binomial shot noise with configurable shot count
- NV decoherence: configurable `T2`, default `800 us`

The simulator should keep fast smoke-test settings available through CLI arguments. The research defaults above should be the defaults used by the Colab notebook.

Each sample should expose:

- `signals`: two traces with shape `(2, 4000)` for research settings
- `taus_us`: matching tau grids
- `heatmap`: one heatmap with shape `(1, 128, 256)`
- `spins`: padded true `(A_z, A_perp)` list in kHz
- `n_spins`: true spin count

The Gaussian heatmap coordinate system should be configurable and saved into checkpoints.

## Model And Training

The model should preserve the SALI two-branch structure:

- One 1D branch processes the `N=32` trace.
- One 1D branch processes the `N=256` trace.
- Branch features are fused and decoded into a 2D heatmap.

Required model change:

- Remove the hard-coded `32 x 64` output assumption.
- Support `128 x 256` as the research output shape.
- Keep smaller output shapes available for smoke tests.
- Use pooling or adaptive pooling so long `4000`-point traces remain practical.

Training should write:

- Best checkpoint containing model state, training arguments, heatmap spec, and best validation score.
- Metric history file, such as CSV or JSONL, containing epoch-level train loss, validation loss, precision, recall, and matched MAE.
- Optional per-threshold evaluation output for plotting threshold sweeps.

The first implementation should use MSE heatmap loss to stay aligned with the current prototype. A BCE/MSE hybrid can be added later if the heatmap peaks are too diffuse. Blob decoding and nearest-neighbor matching should provide the spin-level metrics.

## PyCCE Testing

PyCCE testing should generate random C13 spin baths independently from analytic training data.

For each PyCCE test sample, save:

- Raw `N=32` trace
- Raw `N=256` trace
- Full generated spin table
- Hyperfine tensors
- Projected `(A_z, A_perp)` values in kHz
- Detectable-spin mask
- Ground-truth heatmap for the detectable subset

The main benchmark should score only detectable spins. The full bath remains saved for inspection so the benchmark is fair without hiding the physical context.

## Hyperfine Tensor Projection

Each PyCCE `3 x 3` hyperfine tensor should be projected into model coordinates using the magnetic-field/NV quantization axis.

Default axis:

```text
b_hat = z
Bz = 525 G
```

Projection:

```text
A_z = b_hat^T A b_hat
A_perp = sqrt(||A b_hat||^2 - A_z^2)
```

Output units must be kHz. The implementation should make unit conversion explicit so Hz, rad/s, and kHz are not mixed silently.

## Decomposition Baseline

The baseline is the local repository:

```text
/Users/weitao/Code/python/dqpmodel/cpmg_model
```

The relevant API is:

```python
extract_hyperfine_parameters(tau_us, signal, config=ExtractionConfig(...))
```

For the first comparison:

- Run the baseline on the `N=32` trace only.
- Use `magnetic_field_gauss = 525.0`.
- Compare its returned `SpinEstimate` values with the same matcher used for AI output.

Canonical comparison coordinates are the AI/PyCCE label coordinates:

```text
(A_z, A_perp) = (physical A_par, physical A_perp)
```

Important `cpmg_model` convention:

```text
cpmg_model A = -A_par
cpmg_model B = A_perp
```

Comparison reports must label this convention explicitly. Before metric matching, convert baseline outputs with:

```text
A_z = -A
A_perp = B
```

## Notebook And Colab Workflow

The notebook remains the readable research workflow, but it should call reusable package CLIs/scripts instead of containing hidden one-off logic.

The notebook must contain or display:

1. Raw CPMG traces: `N=32` and `N=256`
2. True spin scatter plot in `(A_z, A_perp)`
3. Ground-truth heatmap label
4. Train/validation loss curve
5. Target heatmap vs predicted heatmap
6. Precision/recall/MAE curve versus epoch or threshold

Implementation should use the `colab-operator` workflow for actual Colab CLI training. Preferred execution is either:

- `colab run --gpu T4 ...` for a self-cleaning training smoke/medium run, or
- a named `colab new -s <name>` session followed by `colab exec -s <name> ...`, with an explicit `colab stop -s <name>` when complete.

The notebook should also remain runnable manually in the Colab UI.

## Evaluation Metrics

Shared metrics should support both AI and baseline outputs:

- True positives
- False positives
- False negatives
- Precision
- Recall
- Matched MAE in kHz
- Optional threshold sweep for precision/recall/MAE
- Optional per-sample diagnostic output for failure analysis

The matcher should operate in `(A_z, A_perp)` space with a configurable maximum matching distance in kHz.

## Testing And Validation

Add focused tests for:

- Heatmap coordinate mapping with the new `[-250, 250] x [2, 250] kHz` range.
- Model forward pass for `128 x 256` output.
- Configurable CPMG tau grids with `0-40 us` and `4000` points.
- Shot-noise path and `T2 = 800 us` decoherence behavior.
- Hyperfine tensor projection into kHz.
- Baseline convention conversion for `A = -A_par`, `B = A_perp`.
- Smoke training and evaluation on a tiny dataset.

Local verification should not require PyCCE or Colab. PyCCE and Colab checks can be opt-in or integration-level.

## Error Handling

The implementation should fail clearly when:

- PyCCE is not installed and PyCCE testing is requested.
- The local `dqpmodel/cpmg_model` path is missing.
- The baseline dependency cannot be imported.
- Hyperfine tensor units are unspecified or unsupported.
- A checkpoint heatmap spec does not match the requested model output shape.
- Colab authentication or accelerator allocation fails.

## Out Of Scope For First Implementation

- Training directly on PyCCE-generated traces.
- Fine-tuning on experimental data.
- Multi-trace baseline decomposition beyond native `N=32`.
- Full pulse-imperfection modeling.
- Automated long-running hyperparameter sweeps.

These can be added after the modular pipeline produces interpretable benchmark results.
