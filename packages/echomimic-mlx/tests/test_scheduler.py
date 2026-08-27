from __future__ import annotations

import mlx.core as mx
import numpy as np
from echomimic_mlx.wan.sampler import FlowEulerDiscreteScheduler


def test_distilled_euler_schedule_uses_requested_indices_and_reaches_zero() -> None:
    scheduler = FlowEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps([1000, 875, 750, 625, 500, 375, 250, 125], shift=5.0)

    sample = mx.ones((2,), dtype=mx.float32)
    model_output = mx.full((2,), 0.25, dtype=mx.float32)
    initial_sigma = float(scheduler.sigmas[0].item())
    for timestep in scheduler.timesteps:
        sample = scheduler.step(model_output, timestep, sample)
    mx.eval(sample)

    np.testing.assert_allclose(np.array(sample), 1.0 - 0.25 * initial_sigma, atol=1e-6)
