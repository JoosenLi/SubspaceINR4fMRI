# Gradient And Parameter Flow

This file focuses on the question:

`Which loss affects which part of the network, and how do gradients flow from the loss back to each parameter group?`

## 1. Parameter Groups In The Model

The trainable parts of `SubspaceINR4fMRI` are:

### A. Shared spatial encoder
File:
- `data/HashINR/models/spatial_encoder_tcnn.py`

Object:
- `TCNNHashGridEncoder`

Role:
- maps voxel coordinates `x` to a learned spatial feature vector

Trainable parameters:
- tiny-cuda-nn hash-grid encoding parameters

### B. Residual head
File:
- `data/HashINR/models/heads.py`

Object:
- `TCNNHead`

Role:
- predicts `I_res(x)` when `model.use_residual=true`

Trainable parameters:
- tiny-cuda-nn MLP weights for residual output

### C. Coefficient head
File:
- `data/HashINR/models/heads.py`

Object:
- `TCNNHead`

Role:
- predicts real-valued coefficient vector `c(x)`

Trainable parameters:
- tiny-cuda-nn MLP weights for coefficient output

### D. Delay head
File:
- `data/HashINR/models/heads.py`

Object:
- `TCNNHead`

Role:
- predicts raw delay, then bounded to `tau_norm(x)`

Trainable parameters:
- tiny-cuda-nn MLP weights for delay output

### E. Temporal basis network
File:
- `data/HashINR/models/temporal_basis.py`

Object:
- `TemporalBasisNetwork`

Role:
- predicts `g(t_norm)` and optionally `dg/dt_norm`

Trainable parameters:
- SIREN hidden layer weights
- output linear layer weights

## 2. Fixed Inputs That Do Not Receive Gradients

These are part of the computation graph as data, but they are not trainable parameters:
- `bundle.background_init`
- `bundle.kspace`
- `bundle.traj`
- `bundle.csm`
- `spatial_coords`
- `time_coords`
- `tau_norm_max`

Important:
- `I_init(x)` is fixed data, not a learned module

## 3. Loss Terms And Their Gradient Targets

## 3.1 Data consistency loss

Definition:
- `L_data = complex_mse(predicted_kspace, measured_kspace)`

Path:

```text
L_data
  -> predicted_kspace
  -> NUFFT backward via adj_op
  -> predicted_image
  -> static + dynamic synthesis
  -> spatial components and temporal basis
  -> model parameters
```

Affected parameter groups:
- spatial encoder
- coefficient head
- temporal basis network
- residual head, if enabled
- delay head, if enabled

Why:
- the image prediction depends on all of these branches
- the NUFFT autograd bridge returns image-domain gradients that are further backpropagated through the synthesis formula

## 3.2 Residual energy loss

Definition:
- `L_res = mean(|I_res(x)|^2)`

Affected parameter groups:
- residual head only
- shared spatial encoder indirectly

Not affected:
- coefficient head
- delay head
- temporal basis network

Why:
- `L_res` is computed directly from `spatial_components.residual`

## 3.3 Tau smoothness loss

Definition:
- finite-difference smoothness on `tau(x)`

Affected parameter groups:
- delay head only
- shared spatial encoder indirectly

Not affected:
- residual head
- coefficient head
- temporal basis network

Why:
- `L_tau` is computed only from the delay field

## 3.4 Temporal basis smoothness loss

Definition:
- `L_g = mean((dg/dt_norm)^2)`

Affected parameter groups:
- temporal basis network only

Not affected:
- spatial encoder
- any spatial head

Why:
- `L_g` is computed only from the temporal derivative tensor

## 4. Branch-Wise Gradient Summary

| Branch | Receives gradient from `L_data` | Receives gradient from `L_res` | Receives gradient from `L_tau` | Receives gradient from `L_g` |
| --- | --- | --- | --- | --- |
| Spatial encoder | yes | yes when residual enabled | yes when delay enabled | no |
| Residual head | yes when enabled | yes when enabled | no | no |
| Coefficient head | yes | no | no | no |
| Delay head | yes when enabled | no | yes when enabled | no |
| Temporal basis network | yes | no | no | yes |

## 5. Detailed Backprop Through Synthesis

## 5.1 Without delay

The dynamic term is:
- `dynamic = einsum(coeff, basis)`

So `L_data` sends gradients to:
- `coeff`
- `basis`

Then:
- gradient on `coeff` flows into:
  - coefficient head
  - shared spatial encoder
- gradient on `basis` flows into:
  - temporal basis network

## 5.2 With delay

The dynamic term is:
- `dynamic = einsum(coeff, basis) - tau * einsum(coeff, basis_derivative)`

So `L_data` sends gradients to:
- `coeff`
- `basis`
- `tau`
- `basis_derivative`

Then:
- `coeff` gradient flows to coefficient head and spatial encoder
- `tau` gradient flows to delay head and spatial encoder
- `basis` and `basis_derivative` gradients flow to the temporal basis network

This is why enabling delay couples the data term to both:
- the delay branch
- the derivative-aware temporal branch

## 6. NUFFT Backward In More Detail

The custom autograd bridge in `data/HashINR/nufft/nufft_utils.py` is the key measurement-model link.

Forward:
- `operator.op(predicted_image_frame)`

Backward:
- `operator.adj_op(grad_output_frame)`

Interpretation:
- the adjoint NUFFT acts as the Jacobian-transpose action needed to bring k-space residual gradients back into image space

Without this bridge, the data-consistency term would not properly train the INR through the measurement model.

## 7. What Changes When Optional Branches Are Disabled

## 7.1 `model.use_residual=false`
- residual head is not instantiated
- `I_res` is not synthesized
- `L_res` is skipped
- no gradients can flow into a residual branch because it does not exist

## 7.2 `model.use_delay=false`
- delay head is not instantiated
- `tau` is not computed
- `dg/dt_norm` is not required for synthesis
- `L_tau` is skipped
- data-consistency gradients only flow through:
  - coefficient head
  - temporal basis values `g(t_norm)`
  - spatial encoder

## 8. How Optimizer Updates Are Applied

File:
- `data/HashINR/training/train_subspace_inr.py`

Sequence:
1. `optimizer.zero_grad(...)`
2. forward pass
3. loss assembly
4. `total_loss.backward()`
5. optional grad clipping
6. `optimizer.step()`

So every trainable submodule receives updates in the same Adam optimizer unless a branch is disabled structurally.

## 9. Practical Interpretation

### The coefficient head
Learns where in space each temporal basis is expressed.

### The temporal basis network
Learns what time courses explain the dynamic signal.

### The delay head
Learns local timing shifts for the shared bases.

### The residual head
Learns static mismatch not already captured by the fixed background.

### The spatial encoder
Is shared, so it becomes a common feature backbone for all enabled spatial heads.

That shared encoder is one of the most important design choices in the current implementation:
- it lets all enabled spatial branches coordinate through one learned spatial representation
- it also means multiple losses can pull on the same backbone in different ways
