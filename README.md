# adaptive-warmup

`adaptive-warmup` estimates a useful batch size and learning rate during the
early part of any PyTorch training run. It is the model-agnostic extraction of
the adaptive warmup originally built for the HeAR distillation trainer.

It combines two measurements:

- **Critical batch size:** two independent minibatch gradients estimate the
  gradient noise-to-signal ratio and the largest statistically useful batch.
- **Directional critical learning rate:** a held-out loss is evaluated along
  the upcoming optimizer direction to find the largest locally non-increasing
  trial step.

The controller filters those noisy measurements, forecasts the end-of-warmup
learning-rate target, grows batch size toward a configurable utility point,
and remembers OOM-derived hardware caps. It never owns your model, optimizer,
data loader, scheduler, or training loop.

## Install

```bash
git clone <repository-url> adaptive-warmup
cd adaptive-warmup
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Minimal integration

Measure after `loss.backward()` and before `optimizer.step()`. The learning-rate
probe uses the gradients of the real upcoming update; the noise probe uses
`torch.autograd.grad` and leaves those real gradients untouched.

```python
import torch
from adaptive_warmup import (
    AdaptiveWarmup,
    WarmupConfig,
    estimate_critical_learning_rate,
    estimate_gradient_noise,
    set_reference_lr,
)

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
warmup = AdaptiveWarmup(
    initial_lr=1e-5,
    initial_batch_size=16,
    config=WarmupConfig(
        warmup_steps=1_000,
        measurement_interval=5,
        max_lr=3e-3,
        max_batch_size=512,
    ),
)

for step in range(max_steps):
    batch = next_batch(warmup.current_batch_size)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_for(batch)
    loss.backward()

    if warmup.should_measure(step):
        probe_a = next_batch(warmup.current_batch_size)
        probe_b = next_batch(warmup.current_batch_size)
        held_out = next_validation_batch()

        batch_estimate = estimate_gradient_noise(
            model.parameters(),
            loss_for,
            probe_a,
            probe_b,
            batch_size=warmup.current_batch_size,
        )
        lr_estimate = estimate_critical_learning_rate(
            lambda: loss_for(held_out),
            model=model,
            optimizer=optimizer,
            current_lr=optimizer.param_groups[0]["lr"],
            previous_estimate=warmup.last_critical_lr,
            max_lr=warmup.config.max_lr,
        )
        recommendation = warmup.observe(
            step,
            critical_lr=lr_estimate,
            critical_batch_size=batch_estimate,
        )
        set_reference_lr(optimizer, recommendation.learning_rate)
        rebuild_loader_if_needed(recommendation.batch_size)

    optimizer.step()
```

If CUDA runs out of memory, call
`warmup.report_oom(step, failed_batch_size=...)`, clear the partial gradients,
and retry with the returned batch size. Save `warmup.state_dict()` beside the
model and optimizer; restore it with `warmup.load_state_dict(...)`.

Run the self-contained synthetic example:

```bash
python examples/train_toy.py
```

## What the recommendations mean

The batch recommendation targets the smallest batch whose mean modeled utility
reaches `batch_target_utility` over a recent window of critical-batch samples.
The default utility is 0.5, which selects the critical batch for a stable single
sample. Set `batch_multiplier > 1` when step-time throughput matters more than
sample efficiency.

The LR estimate is local to the current parameters, gradients, optimizer state,
and held-out batch. It is not a global convergence guarantee or an exact Hessian
eigenvalue. The default controller applies a 0.8 safety factor, freezes new LR
target measurements halfway through warmup, and gradually reaches that frozen
target while batch measurements continue.

## Supported training setups

- Any differentiable PyTorch model and scalar loss closure.
- Built-in optimizer-direction previews for `torch.optim.SGD`, `Adam`, and
  `AdamW`, including momentum, AMSGrad, weight decay, and multiple LR groups.
- Custom optimizers through an explicit `OptimizerDirection`.
- CPU or CUDA, FP32 or autocast. With `GradScaler`, unscale the optimizer before
  the LR probe.

Important boundaries:

- Use independent, representative batches for the gradient-noise probe.
- Keep the held-out LR closure deterministic. The library temporarily switches
  the supplied model to evaluation mode and restores all module mode flags.
- The gradient-noise closure runs in its current mode; models with mutable
  buffers such as BatchNorm may need a project-specific buffer-preservation
  wrapper.
- In distributed training, run identical probes on every rank or probe on one
  rank and broadcast the recommendation at an explicit synchronization point.
- Probing costs two backward passes plus several no-grad held-out forwards, so
  use a cadence such as every 5–20 steps rather than every step.

See [the algorithm note](docs/algorithm.md) for equations, integration order,
and assumptions.

## Background

The batch estimator follows the gradient-noise-scale motivation in
[McCandlish et al., *An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162).
The held-out directional probe is a deliberately simple local line search; for
a probabilistic stochastic line-search treatment, see
[Mahsereci and Hennig](https://www.jmlr.org/papers/v18/17-049.html).

MIT licensed.
