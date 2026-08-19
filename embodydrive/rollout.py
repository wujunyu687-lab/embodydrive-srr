"""Latent flow-matching rollout utilities shared by SRR and TRD."""

import math

import torch
import torch.nn.functional as F


def adaptive_transition_stabilize(
    future_latents,
    history_latents,
    reference_frames=4,
    max_motion_ratio=2.0,
):
    """Cap only anomalous latent jumps relative to recent history motion.

    The stabilizer is intentionally conservative: normal motion is untouched;
    when a newly generated frame jumps much farther than the recent history,
    its whole transition vector is scaled to the allowed motion magnitude.
    """
    if max_motion_ratio <= 0.0:
        raise ValueError("max_motion_ratio must be positive")
    if history_latents.shape[2] < 2:
        return future_latents
    count = min(int(reference_frames), history_latents.shape[2] - 1)
    recent = history_latents[:, :, -count:]
    previous = history_latents[:, :, -(count + 1) : -1]
    reference = (recent.float() - previous.float()).abs().mean()
    jump = (future_latents.float() - history_latents[:, :, -1:].float()).abs().mean()
    limit = reference * float(max_motion_ratio)
    if jump <= limit or jump <= 1e-8:
        return future_latents
    scale = (limit / jump).to(future_latents.dtype)
    return history_latents[:, :, -1:] + (future_latents - history_latents[:, :, -1:]) * scale


def velocity_transition_blend(
    future_latents,
    history_latents,
    blend=0.0,
    reference_frames=4,
):
    """Blend a prediction toward a short-term constant-velocity continuation.

    The ordinary ``latent_blend`` pulls a prediction toward the last frame and
    therefore damps motion.  This variant preserves the recent direction of
    motion while suppressing a discontinuity at the history/prediction seam.
    It is useful as a deployment-time diagnostic and is differentiable, so it
    can also be used by a future consistency objective.
    """
    blend = float(blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if blend == 0.0 or history_latents.shape[2] < 2:
        return future_latents
    count = min(int(reference_frames), history_latents.shape[2] - 1)
    recent = history_latents[:, :, -count:]
    previous = history_latents[:, :, -(count + 1):-1]
    velocity = (recent.float() - previous.float()).mean(dim=2, keepdim=True)
    steps = torch.arange(
        1, future_latents.shape[2] + 1,
        device=future_latents.device,
        dtype=future_latents.dtype,
    ).view(1, 1, -1, 1, 1)
    expected = history_latents[:, :, -1:] + velocity.to(future_latents.dtype) * steps
    return (1.0 - blend) * future_latents + blend * expected


def select_continuous_candidate(
    candidates,
    history_latents,
    reference_frames=4,
):
    """Select the sampled future with the most plausible local transition.

    ``candidates`` is ``[ensemble, batch, channels, time, height, width]``.
    The score is the latent L1 distance from a short constant-velocity
    continuation of the accepted history.  Unlike averaging, selection keeps
    the result on one sampled trajectory and is therefore less likely to land
    between distinct motion modes.
    """
    if candidates.ndim != 6:
        raise ValueError("candidates must have shape [ensemble,batch,channels,time,height,width]")
    if candidates.shape[0] <= 0:
        raise ValueError("candidate ensemble must be non-empty")
    if history_latents.shape[2] < 2:
        return candidates[0]
    count = min(int(reference_frames), history_latents.shape[2] - 1)
    recent = history_latents[:, :, -count:]
    previous = history_latents[:, :, -(count + 1):-1]
    velocity = (recent.float() - previous.float()).mean(dim=2, keepdim=True)
    steps = torch.arange(
        1, candidates.shape[3] + 1,
        device=candidates.device,
        dtype=candidates.dtype,
    ).view(1, 1, 1, -1, 1, 1)
    expected = history_latents[:, :, -1:].unsqueeze(0) + velocity.to(candidates.dtype).unsqueeze(0) * steps
    scores = (candidates.float() - expected.float()).abs().mean(dim=(2, 3, 4, 5))
    selected = scores.argmin(dim=0)
    batch_index = torch.arange(candidates.shape[1], device=candidates.device)
    return candidates[selected, batch_index]


def history_condition(history_latents, total_frames):
    """Pack clean history latents and a binary history mask for the adapter."""
    history_frames = history_latents.shape[2]
    if history_frames >= total_frames:
        raise ValueError("history must be shorter than the generated window")
    zeros = torch.zeros(
        history_latents.shape[0],
        history_latents.shape[1],
        total_frames - history_frames,
        history_latents.shape[3],
        history_latents.shape[4],
        device=history_latents.device,
        dtype=history_latents.dtype,
    )
    mask = torch.zeros(
        history_latents.shape[0],
        1,
        total_frames,
        history_latents.shape[3],
        history_latents.shape[4],
        device=history_latents.device,
        dtype=history_latents.dtype,
    )
    mask[:, :, :history_frames] = 1
    return torch.cat([torch.cat([history_latents, zeros], dim=2), mask], dim=1)


def _seq_len(model, latents):
    base_model = getattr(model, "module", model)
    patch_size = tuple(base_model.config.patch_size)
    return math.ceil(
        latents.shape[2] * latents.shape[3] * latents.shape[4] / math.prod(patch_size)
    )


def euler_sample_chunk(
    model,
    history_latents,
    action_conditions,
    proprio_conditions,
    future_frames,
    num_steps,
    noise=None,
    context=None,
):
    """Generate one future latent chunk while keeping history fixed.

    The model predicts flow from sigma=1 to sigma=0. The returned tensor only
    contains the newly generated future frames; gradients are preserved unless
    the caller wraps this function in ``torch.no_grad``.
    """
    batch_size = history_latents.shape[0]
    total_frames = history_latents.shape[2] + future_frames
    if action_conditions is not None and action_conditions.shape[1] != total_frames:
        raise ValueError(
            "action condition length must equal history + future frames: "
            f"history={history_latents.shape[2]} future={future_frames} "
            f"action={action_conditions.shape[1]}"
        )
    if proprio_conditions is not None and proprio_conditions.shape[1] != total_frames:
        raise ValueError(
            "proprio condition length must equal history + future frames: "
            f"history={history_latents.shape[2]} future={future_frames} "
            f"proprio={proprio_conditions.shape[1]}"
        )
    if context is None:
        context = [
            torch.zeros(512, 4096, device=history_latents.device, dtype=history_latents.dtype)
            for _ in range(batch_size)
        ]
    state_shape = (
        batch_size,
        history_latents.shape[1],
        future_frames,
        history_latents.shape[3],
        history_latents.shape[4],
    )
    state = torch.randn(state_shape, device=history_latents.device, dtype=history_latents.dtype) if noise is None else noise
    if tuple(state.shape) != state_shape:
        raise ValueError(f"noise shape {tuple(state.shape)} != {state_shape}")
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=state.device, dtype=state.dtype)
    for index in range(num_steps):
        window = torch.cat([history_latents, state], dim=2)
        conditions = {"history": history_condition(history_latents, total_frames)}
        if action_conditions is not None:
            conditions["action"] = action_conditions
        if proprio_conditions is not None:
            conditions["proprio"] = proprio_conditions
        sigma = sigmas[index].expand(batch_size)
        prediction = model(
            window,
            t=sigma,
            context=context,
            seq_len=_seq_len(model, window),
            num_views=1,
            dtype=state.dtype,
            crossview_attn_type="full",
            additional_conditions=conditions,
        )
        state = state + (sigmas[index + 1] - sigmas[index]) * prediction[:, :, -future_frames:]
    return state


def predict_flow_with_history(
    model,
    history_latents,
    future_latents,
    action_conditions,
    proprio_conditions,
    sigma,
    context=None,
):
    """Evaluate a model on a fixed history/future window for distillation."""
    batch_size = history_latents.shape[0]
    total_frames = history_latents.shape[2] + future_latents.shape[2]
    if action_conditions is not None and action_conditions.shape[1] != total_frames:
        raise ValueError("action condition length must equal history + future frames")
    if proprio_conditions is not None and proprio_conditions.shape[1] != total_frames:
        raise ValueError("proprio condition length must equal history + future frames")
    if context is None:
        context = [
            torch.zeros(512, 4096, device=history_latents.device, dtype=history_latents.dtype)
            for _ in range(batch_size)
        ]
    window = torch.cat([history_latents, future_latents], dim=2)
    conditions = {"history": history_condition(history_latents, total_frames)}
    if action_conditions is not None:
        conditions["action"] = action_conditions
    if proprio_conditions is not None:
        conditions["proprio"] = proprio_conditions
    return model(
        window,
        t=sigma,
        context=context,
        seq_len=_seq_len(model, window),
        num_views=1,
        dtype=window.dtype,
        crossview_attn_type="full",
        additional_conditions=conditions,
    )


def continuation_flow_loss(
    model,
    latents,
    action_conditions,
    proprio_conditions,
    history_frames,
    endpoint_probability=0.0,
    sigma_mode="random",
    sigma_steps=0,
):
    """Flow-matching loss for a clean history and noisy future window."""
    if history_frames <= 0 or history_frames >= latents.shape[2]:
        raise ValueError("history_frames must be smaller than the latent window")
    batch_size = latents.shape[0]
    dtype = latents.dtype
    if sigma_mode == "euler" and int(sigma_steps) > 0:
        grid = torch.linspace(
            1.0,
            0.0,
            int(sigma_steps) + 1,
            device=latents.device,
            dtype=dtype,
        )[:-1]
        sigma = grid[torch.randint(len(grid), (batch_size,), device=latents.device)]
    else:
        sigma = torch.rand(batch_size, device=latents.device, dtype=dtype)
    endpoint_probability = float(max(0.0, min(1.0, endpoint_probability)))
    if endpoint_probability > 0.0:
        endpoint = torch.rand(batch_size, device=latents.device) < endpoint_probability
        endpoint_sigma = torch.randint(
            0,
            2,
            (batch_size,),
            device=latents.device,
            dtype=torch.int64,
        ).to(dtype)
        sigma = torch.where(endpoint, endpoint_sigma, sigma)
    sigma_view = sigma.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(latents)
    noisy_future = (1.0 - sigma_view) * latents + sigma_view * noise
    model_input = torch.cat(
        [latents[:, :, :history_frames], noisy_future[:, :, history_frames:]], dim=2
    )
    conditions = {"history": history_condition(latents[:, :, :history_frames], latents.shape[2])}
    if action_conditions is not None:
        conditions["action"] = action_conditions
    if proprio_conditions is not None:
        conditions["proprio"] = proprio_conditions
    context = [
        torch.zeros(512, 4096, device=latents.device, dtype=dtype)
        for _ in range(batch_size)
    ]
    prediction = model(
        model_input,
        t=sigma,
        context=context,
        seq_len=_seq_len(model, model_input),
        num_views=1,
        dtype=dtype,
        crossview_attn_type="full",
        additional_conditions=conditions,
    )
    target = noise - latents
    future_mask = torch.zeros_like(target)
    future_mask[:, :, history_frames:] = 1
    # Average only over supervised future tokens.  Averaging over the whole
    # history+future window would dilute the gradient by history_frames / total
    # frames (21x for history=20, chunk=1), making the reported loss look small
    # while barely training the autoregressive recovery target.
    error = ((prediction - target).float() * future_mask.float()).square()
    return error.sum() / future_mask.float().sum().clamp_min(1.0)
