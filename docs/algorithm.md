# Algorithm

## 1. Gradient-noise batch estimate

Draw two independent minibatches of the same size `B` and compute sampled
gradient vectors `g1` and `g2`. The library uses:

```text
noise  = 0.5 * ||g1 - g2||²
signal = ||0.5 * (g1 + g2)||²
noise-to-signal = noise / max(signal, epsilon)
critical batch = B * noise-to-signal
```

Only a deterministic prefix of the parameter vector is required; the default
cap is 200,000 gradient elements. `torch.autograd.grad` keeps the real training
gradients in `.grad` intact.

Raw critical-batch measurements are multiplicatively outlier-clamped around an
EMA and kept in a short window. For a candidate batch `b` and measurements
`Bcrit_i`, modeled large-batch utility is:

```text
mean_i b / (b + Bcrit_i)
```

Binary search selects the smallest `b` meeting the configured utility target.
The result is multiplied, rounded down, capped by user and OOM limits, and only
allowed to grow during normal warmup.

## 2. Directional learning-rate estimate

After the real training backward pass, the library previews the next optimizer
direction `d`, including momentum/moment state and weight decay. For optimizers
with multiple parameter-group LRs, `d` preserves their LR ratios relative to a
single reference LR.

On one fixed held-out batch it evaluates:

```text
L(learning_rate) = held_out_loss(parameters - learning_rate * d)
```

Starting from the current or previous estimate, it doubles the trial LR until
the loss becomes non-finite or exceeds the base loss by `loss_tolerance`. A
binary search then returns the largest accepted LR. Parameter changes are
reversed in a `finally` block and optimizer state is never changed.

This is an empirical, local directional boundary. It should not be interpreted
as the globally stable LR or an exact inverse sharpness measurement.

## 3. Warmup controller

The controller:

1. outlier-clamps and exponentially smooths LR samples;
2. fits a Theil–Sen slope to recent log critical-LR measurements;
3. forecasts the terminal critical LR and applies a safety factor;
4. caps that forecast by the current safe sample and first observed boundary;
5. freezes the LR target at `lr_freeze_fraction` of warmup;
6. ramps toward the frozen target over remaining measurement updates;
7. continues critical-batch measurements and batch growth through warmup end.

The default first 10% of warmup limits each measured LR increase to 10%. OOM
reports reduce batch size with headroom and create a persistent hardware cap.

## Integration order

For a zero-based training step:

```text
real forward -> real backward -> optional probes -> apply recommendation
             -> optimizer step -> next batch
```

The LR recommendation applies to the optimizer step whose gradients supplied
the direction. A changed batch size applies when constructing the next batch.

With AMP `GradScaler`, call `scaler.unscale_(optimizer)` before the LR probe so
the preview sees unscaled gradients. With gradient accumulation, run the LR
probe after the final microbatch; define `batch_size` for the noise probe as
the number of examples represented by each independently computed probe
gradient.
