"""Scheduled Rollout Recovery (SRR) teacher training for EmbodyDrive."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embodydrive.droid_dataset import DroidVideoActionDataset
from embodydrive.offline_dataset import DroidWristLatentDataset
from embodydrive.rollout import (
    continuation_flow_loss,
    euler_sample_chunk,
    velocity_transition_blend,
)
from embodydrive.train_g0 import build_transformer, downsample_temporal
from videox_fun.models.wan_vae import AutoencoderKLWan


def load_model_weights(model, checkpoint):
    """Load only model weights from a G0/SRR checkpoint.

    SRR must start with a fresh optimizer. ``accelerator.load_state`` would
    also restore the source optimizer moments and can silently override the
    requested learning-rate schedule.
    """
    path = Path(checkpoint)
    weight_path = path / "model.safetensors" if path.is_dir() else path
    if not weight_path.is_file():
        raise FileNotFoundError(f"model weights not found: {weight_path}")
    from safetensors.torch import load_file

    state = load_file(str(weight_path), device="cpu")
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "action.token_projection.weight",
        "action.token_projection.bias",
        "proprio.modulation_projection.weight",
        "proprio.modulation_projection.bias",
        "action.circular_embedding.weight",
        "action.circular_embedding.bias",
        "proprio.circular_embedding.weight",
        "proprio.circular_embedding.bias",
    }
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible model checkpoint: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="EmbodyDrive SRR teacher training")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--offline-root", default=None, help="Offline wrist latent cache")
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Logical starting step for model-only continuation; preserves rollout curriculum position",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--history-latents", type=int, default=1)
    parser.add_argument("--rollout-chunk-latents", type=int, default=4)
    parser.add_argument("--rollout-depth", type=int, default=3)
    parser.add_argument("--min-rollout-depth", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument(
        "--sigma-mode",
        choices=("random", "euler"),
        default="random",
        help="Flow sigma sampling distribution; euler matches the rollout solver grid",
    )
    parser.add_argument(
        "--endpoint-probability",
        type=float,
        default=0.0,
        help="Optional probability of sampling sigma exactly at 0 or 1",
    )
    parser.add_argument(
        "--self-forcing",
        action="store_true",
        help="Train every autoregressive position using the model's own detached history",
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=2500,
        help="Steps over which self-forcing rollout depth grows to rollout-depth",
    )
    parser.add_argument(
        "--rollout-loss-decay",
        type=float,
        default=1.0,
        help="Optional weight decay from early to late self-forcing positions",
    )
    parser.add_argument(
        "--rollout-position-growth",
        type=float,
        default=1.0,
        help=(
            "Geometric weight growth from early to late self-forcing positions; "
            "values above 1 emphasize long-horizon recovery"
        ),
    )
    parser.add_argument(
        "--endpoint-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Direct latent endpoint loss weight for a differentiable Euler rollout; "
            "zero disables this auxiliary objective"
        ),
    )
    parser.add_argument(
        "--endpoint-loss-positions",
        choices=("first", "last", "both", "all"),
        default="first",
        help="Which self-forcing positions receive the direct endpoint loss",
    )
    parser.add_argument(
        "--endpoint-ensemble-size",
        type=int,
        default=1,
        help="Number of independent Euler endpoint samples averaged by endpoint loss",
    )
    parser.add_argument(
        "--endpoint-consistency-weight",
        type=float,
        default=0.0,
        help="Optional penalty on disagreement between endpoint samples",
    )
    parser.add_argument(
        "--self-forcing-position",
        type=int,
        default=-1,
        help=(
            "Only backpropagate the selected self-forcing entry; -1 keeps the "
            "original all-position objective, while a nonnegative index targets "
            "a specific late rollout position"
        ),
    )
    parser.add_argument(
        "--teacher-forced-random-position",
        action="store_true",
        help=(
            "Train one randomly selected target position using its clean GT history; "
            "useful for long-window single-step stabilization"
        ),
    )
    parser.add_argument(
        "--self-forcing-random-position",
        action="store_true",
        help=(
            "Randomly choose a rollout position, generate its history with the "
            "model itself, and train only that contaminated position"
        ),
    )
    parser.add_argument(
        "--self-forcing-random-min-position",
        type=int,
        default=0,
        help="Minimum zero-based rollout position sampled by self-forcing-random-position",
    )
    parser.add_argument(
        "--self-forcing-ensemble-size",
        type=int,
        default=1,
        help=(
            "Number of independent samples averaged while constructing detached "
            "self-forcing histories"
        ),
    )
    parser.add_argument(
        "--self-forcing-velocity-blend",
        type=float,
        default=0.0,
        help="Deployment-style velocity blend for detached self-forcing histories",
    )
    parser.add_argument(
        "--self-forcing-rollout-steps",
        type=int,
        default=0,
        help=(
            "Euler steps used only to generate detached self-forcing histories; "
            "0 reuses rollout-steps"
        ),
    )
    parser.add_argument("--boundary-blend-max", type=float, default=0.25)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0,
                        help="Weight of clean G0 continuation loss used to preserve short-horizon quality")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--condition-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiply the learning rate of action/proprio adapter parameters",
    )
    parser.add_argument(
        "--train-condition-only",
        action="store_true",
        help="Freeze the Wan backbone and update only action/proprio/history adapters",
    )
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def encode_batch(batch, vae, accelerator):
    if "latents" in batch:
        latents = batch["latents"].to(accelerator.device, dtype=torch.bfloat16)
        latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        action = downsample_temporal(
            batch["action"].to(accelerator.device, dtype=torch.bfloat16),
            latents.shape[2], circular_dims=(3, 4, 5),
        )
        proprio = downsample_temporal(
            batch["proprio"].to(accelerator.device, dtype=torch.bfloat16),
            latents.shape[2], circular_dims=(10, 11, 12),
        )
        return latents, action, proprio
    video = batch["video"].to(accelerator.device, dtype=torch.bfloat16)
    video = video.permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad(), accelerator.autocast():
        latents = vae.encode(video).latent_dist.sample()
    action = downsample_temporal(
        batch["action"].to(accelerator.device, dtype=torch.bfloat16),
        latents.shape[2], circular_dims=(3, 4, 5),
    )
    proprio = downsample_temporal(
        batch["proprio"].to(accelerator.device, dtype=torch.bfloat16),
        latents.shape[2], circular_dims=(10, 11, 12),
    )
    return latents, action, proprio


def flow_loss_kwargs(args):
    return {
        "sigma_mode": args.sigma_mode,
        "sigma_steps": args.rollout_steps,
        "endpoint_probability": args.endpoint_probability,
    }


def ensemble_endpoint_losses(
    model,
    history,
    action_window,
    proprio_window,
    target,
    args,
):
    """Endpoint loss aligned with inference-time multi-sample averaging.

    Averaging per-sample L1 losses does not train the quantity used by the
    stable inference path.  Average the predicted endpoints first, then
    compare that mean to GT.  The optional consistency term additionally
    discourages excessive disagreement between noise samples.
    """
    predictions = torch.stack(
        [
            euler_sample_chunk(
                model,
                history,
                action_window,
                proprio_window,
                args.rollout_chunk_latents,
                args.rollout_steps,
            )
            for _ in range(args.endpoint_ensemble_size)
        ],
        dim=0,
    )
    mean_prediction = predictions.mean(dim=0)
    endpoint = (mean_prediction.float() - target.float()).abs().mean()
    consistency = (predictions.float() - mean_prediction.float()).abs().mean()
    return endpoint.to(dtype=target.dtype), consistency.to(dtype=target.dtype)


def detached_rollout_future(model, history, action_window, proprio_window, args):
    """Generate detached history state with deployment-like sampling."""
    history_steps = int(args.self_forcing_rollout_steps or args.rollout_steps)
    if history_steps <= 0:
        raise ValueError("self-forcing-rollout-steps must be positive")
    predictions = [
        euler_sample_chunk(
            model,
            history,
            action_window,
            proprio_window,
            args.rollout_chunk_latents,
            history_steps,
        )
        for _ in range(args.self_forcing_ensemble_size)
    ]
    future = torch.stack(predictions, dim=0).mean(dim=0)
    if args.self_forcing_velocity_blend > 0.0:
        future = velocity_transition_blend(
            future,
            history,
            args.self_forcing_velocity_blend,
            reference_frames=min(4, history.shape[2] - 1),
        )
    return future


def polluted_history(model, gt_latents, actions, proprio, args, rollout_depth=None, blend_ratio=0.0):
    """Roll out chunks from the clean prefix and return a detached history."""
    history_frames = args.history_latents
    chunk_frames = args.rollout_chunk_latents
    available_depth = (gt_latents.shape[2] - history_frames) // chunk_frames - 1
    if available_depth < 1:
        raise ValueError("clip is too short for the requested SRR rollout depth")
    depth = min(args.rollout_depth if rollout_depth is None else rollout_depth, available_depth)
    history = gt_latents[:, :, :history_frames]
    with torch.no_grad():
        was_training = model.training
        model.eval()
        for rollout_index in range(depth):
            # Each next window advances by the generated chunk, while retaining
            # the last history frames as context.
            start = rollout_index * chunk_frames
            end = start + history_frames + chunk_frames
            if chunk_frames != args.rollout_chunk_latents:
                raise ValueError("polluted history chunk does not match rollout chunk")
            future = detached_rollout_future(
                model,
                history,
                actions[:, start:end],
                proprio[:, start:end],
                args,
            )
            # Keep the previous history together with the generated future
            # before taking the tail. This also works when chunk < history.
            history = torch.cat([history, future], dim=2)[:, :, -history_frames:]
    if was_training:
        model.train()
    target_start = history_frames + depth * chunk_frames
    blend_ratio = float(max(0.0, min(1.0, blend_ratio)))
    if blend_ratio > 0.0:
        clean_boundary = gt_latents[:, :, target_start - history_frames : target_start]
        history = (1.0 - blend_ratio) * history + blend_ratio * clean_boundary
    return history.detach(), target_start


def self_forcing_histories(
    model, gt_latents, actions, proprio, args, rollout_depth=None, blend_ratio=0.0
):
    """Build detached histories for every autoregressive training position.

    The rollout is intentionally done without autograd: retaining all Euler
    solver graphs for a full episode would be prohibitively expensive for the
    1.4B-parameter transformer.  Each generated history is nevertheless fed
    back into a separate supervised flow-matching loss below, which trains the
    model on the same state distribution used by inference.
    """
    history_frames = int(args.history_latents)
    chunk_frames = int(args.rollout_chunk_latents)
    total_frames = gt_latents.shape[2]
    available_depth = (total_frames - history_frames) // chunk_frames
    if available_depth < 1:
        raise ValueError("clip is too short for self-forcing")
    depth = min(
        int(args.rollout_depth if rollout_depth is None else rollout_depth),
        available_depth,
    )

    history = gt_latents[:, :, :history_frames]
    entries = []
    was_training = model.training
    with torch.no_grad():
        model.eval()
        for rollout_index in range(depth):
            target_start = history_frames + rollout_index * chunk_frames
            entry_history = history.detach()
            # Match the inference-time boundary stabilization used by
            # polluted_history: early training sees a small clean boundary
            # component, which is annealed away as self-forcing becomes more
            # reliable.  Keep the rollout chain itself fully self-forced; the
            # blend is applied only to the supervised history presented at
            # this target position.
            if blend_ratio > 0.0:
                clean_boundary = gt_latents[
                    :, :, target_start - history_frames : target_start
                ]
                entry_history = (
                    (1.0 - blend_ratio) * entry_history
                    + blend_ratio * clean_boundary
                ).detach()
            entries.append((entry_history, target_start))
            if rollout_index + 1 >= depth:
                continue
            action_window = actions[
                :, target_start - history_frames : target_start + chunk_frames
            ]
            proprio_window = proprio[
                :, target_start - history_frames : target_start + chunk_frames
            ]
            if chunk_frames != args.rollout_chunk_latents:
                raise ValueError("self-forcing chunk does not match rollout chunk")
            future = detached_rollout_future(
                model, history, action_window, proprio_window, args
            )
            history = torch.cat([history, future], dim=2)[:, :, -history_frames:]
    if was_training:
        model.train()
    return entries


def scheduled_rollout(args, global_step):
    """Curriculum from one-step recovery to the requested long rollout."""
    max_depth = max(1, args.rollout_depth)
    min_depth = max(1, min(args.min_rollout_depth, max_depth))
    curriculum_steps = int(getattr(args, "curriculum_steps", 0))
    schedule_horizon = curriculum_steps if curriculum_steps > 0 else args.max_steps
    progress = 0.0 if schedule_horizon <= 1 else min(
        1.0, global_step / (schedule_horizon - 1)
    )
    depth = round(min_depth + progress * (max_depth - min_depth))
    # Early training should stay close to the data manifold; gradually remove
    # the clean boundary blend as the model learns to recover its own errors.
    blend = (1.0 - progress) * max(0.0, min(1.0, args.boundary_blend_max))
    return max(min_depth, depth), blend


def save_checkpoint(accelerator, output_dir, step):
    path = Path(output_dir) / f"checkpoint-{step:08d}"
    # All ranks must enter checkpointing together.  Without the leading
    # barrier, non-main ranks can start the next DDP iteration while rank 0 is
    # materializing and writing the 1.4B-parameter state dict.  That races the
    # next collective and can leave every worker spinning after a checkpoint.
    accelerator.wait_for_everyone()
    accelerator.save_state(str(path))
    # Do not let any rank begin the next forward/backward pass until every
    # rank has finished writing its state (including random_states_N.pkl).
    accelerator.wait_for_everyone()
    accelerator.print(f"saved checkpoint: {path}")


def main():
    args = parse_args()
    if not args.offline_root and not args.data_root:
        raise ValueError("provide --offline-root or --data-root")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SRR training")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.start_step < 0 or args.start_step >= args.max_steps:
        raise ValueError("start-step must be non-negative and smaller than max-steps")
    if args.rollout_depth <= 0 or args.min_rollout_depth <= 0:
        raise ValueError("rollout depths must be positive")
    if args.min_rollout_depth > args.rollout_depth:
        raise ValueError("min-rollout-depth cannot exceed rollout-depth")
    if args.clean_loss_weight < 0:
        raise ValueError("clean-loss-weight must be non-negative")
    if not 0.0 <= args.endpoint_probability <= 1.0:
        raise ValueError("endpoint-probability must be in [0, 1]")
    if args.rollout_loss_decay <= 0 or args.rollout_loss_decay > 1:
        raise ValueError("rollout-loss-decay must be in (0, 1]")
    if args.rollout_position_growth <= 0:
        raise ValueError("rollout-position-growth must be positive")
    if args.endpoint_loss_weight < 0:
        raise ValueError("endpoint-loss-weight must be non-negative")
    if args.endpoint_ensemble_size <= 0:
        raise ValueError("endpoint-ensemble-size must be positive")
    if args.self_forcing_ensemble_size <= 0:
        raise ValueError("self-forcing-ensemble-size must be positive")
    if args.endpoint_consistency_weight < 0:
        raise ValueError("endpoint-consistency-weight must be non-negative")
    if not 0.0 <= args.self_forcing_velocity_blend <= 1.0:
        raise ValueError("self-forcing-velocity-blend must be in [0, 1]")
    if args.teacher_forced_random_position and (args.self_forcing or args.self_forcing_random_position):
        raise ValueError("teacher-forced random position cannot be combined with self-forcing")
    if args.self_forcing and args.self_forcing_random_position:
        raise ValueError("choose either self-forcing or self-forcing-random-position")
    if args.self_forcing_random_min_position < 0:
        raise ValueError("self-forcing-random-min-position must be non-negative")
    if not args.dry_run and not args.init_checkpoint:
        raise ValueError("formal SRR training requires --init-checkpoint from G0")
    set_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision="bf16",
        project_dir=args.output_dir,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        with open(Path(args.output_dir) / "training_args.json", "w") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)

    if args.offline_root:
        stats_path = args.stats_path or str(Path(args.offline_root) / "stats.json")
        dataset = DroidWristLatentDataset(
            args.offline_root,
            frames=args.frames,
            split="train",
            max_episodes=args.max_episodes,
            seed=args.seed,
            include_video=False,
            stats_path=stats_path,
        )
    else:
        dataset = DroidVideoActionDataset(
            args.data_root,
            frames=args.frames,
            image_size=args.image_size,
            max_shards=args.max_shards,
            max_episodes=args.max_episodes,
            seed=args.seed,
            random_crop=True,
            include_proprio=True,
        )
    if args.offline_root:
        latent_window = (args.frames - 1) // 4 + 1
        effective_max_position = min(
            args.rollout_depth, latent_window - args.history_latents
        ) - 1
        if args.self_forcing_random_position and effective_max_position < args.self_forcing_random_min_position:
            raise ValueError(
                "frames/history leave no valid self-forcing position: "
                f"frames={args.frames}, latent_window={latent_window}, "
                f"history={args.history_latents}, max_position={effective_max_position}, "
                f"min_position={args.self_forcing_random_min_position}"
            )
        accelerator.print(
            "offline rollout coverage: "
            f"latent_window={latent_window}, max_position={effective_max_position}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=bool(args.offline_root),
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path,
        additional_kwargs={"temporal_compression_ratio": 4, "spatial_compression_ratio": 8},
    ).to(accelerator.device, dtype=torch.bfloat16).eval()
    vae.requires_grad_(False)
    model = build_transformer(args.model_root, True, True, torch.bfloat16)
    if args.init_checkpoint:
        load_model_weights(model, args.init_checkpoint)
        accelerator.print(f"loaded model-only initialization weights from {args.init_checkpoint}")
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if args.train_condition_only:
        for name, parameter in model.named_parameters():
            if not name.startswith(("action.", "proprio.", "history.")):
                parameter.requires_grad_(False)
        accelerator.print("training only action/proprio/history conditioning parameters")
    condition_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("action.", "proprio.", "history.")):
            condition_params.append(parameter)
        else:
            backbone_params.append(parameter)
    optimizer_groups = []
    if backbone_params:
        optimizer_groups.append({"params": backbone_params, "lr": args.learning_rate})
    if condition_params:
        optimizer_groups.append(
            {
                "params": condition_params,
                "lr": args.learning_rate * args.condition_lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if args.dry_run:
        batch = next(iter(loader))
        gt, actions, proprio = encode_batch(batch, vae, accelerator)
        endpoint_loss = torch.zeros((), device=gt.device, dtype=gt.dtype)
        endpoint_consistency = torch.zeros((), device=gt.device, dtype=gt.dtype)
        if args.self_forcing:
            entries = self_forcing_histories(
                model, gt, actions, proprio, args, rollout_depth=args.rollout_depth
            )
            weights = [
                args.rollout_loss_decay ** i * args.rollout_position_growth ** i
                for i in range(len(entries))
            ]
            weight_sum = float(sum(weights))
            values = []
            selected_entries = entries if args.self_forcing_position < 0 else [entries[args.self_forcing_position]]
            selected_weights = weights if args.self_forcing_position < 0 else [1.0]
            for (history, target_start), weight in zip(selected_entries, selected_weights):
                target = gt[:, :, target_start : target_start + args.rollout_chunk_latents]
                window = torch.cat([history, target], dim=2)
                values.append(
                    continuation_flow_loss(
                        model,
                        window,
                        actions[:, target_start - args.history_latents : target_start + args.rollout_chunk_latents],
                        proprio[:, target_start - args.history_latents : target_start + args.rollout_chunk_latents],
                        args.history_latents,
                        **flow_loss_kwargs(args),
                    )
                )
            srr_loss = sum(value * (weight / weight_sum) for value, weight in zip(values, weights))
            actual_depth = len(entries)
            if args.endpoint_loss_weight > 0:
                endpoint_indices = [0] if args.endpoint_loss_positions == "first" else [-1]
                if args.endpoint_loss_positions == "both":
                    endpoint_indices = [0, -1]
                if args.endpoint_loss_positions == "all":
                    endpoint_indices = list(range(len(entries)))
                endpoint_values = []
                for entry_index in endpoint_indices:
                    history, target_start = entries[entry_index]
                    target = gt[:, :, target_start : target_start + args.rollout_chunk_latents]
                    action_window = actions[
                        :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                    ]
                    proprio_window = proprio[
                        :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                    ]
                    endpoint_value, _ = ensemble_endpoint_losses(
                        model, history, action_window, proprio_window, target, args
                    )
                    endpoint_values.append(float(endpoint_value.detach().cpu()))
                    del endpoint_value
                endpoint_loss = torch.tensor(
                    sum(endpoint_values) / max(1, len(endpoint_values)),
                    device=gt.device,
                    dtype=gt.dtype,
                )
        else:
            history, target_start = polluted_history(model, gt, actions, proprio, args)
            target = gt[:, :, target_start : target_start + args.rollout_chunk_latents]
            window = torch.cat([history, target], dim=2)
            srr_loss = continuation_flow_loss(
                model,
                window,
                actions[:, target_start - args.history_latents : target_start + args.rollout_chunk_latents],
                proprio[:, target_start - args.history_latents : target_start + args.rollout_chunk_latents],
                args.history_latents,
                **flow_loss_kwargs(args),
            )
            actual_depth = 1
        clean_window = gt[:, :, : args.history_latents + args.rollout_chunk_latents]
        clean_loss = continuation_flow_loss(
            model,
            clean_window,
            actions[:, : args.history_latents + args.rollout_chunk_latents],
            proprio[:, : args.history_latents + args.rollout_chunk_latents],
            args.history_latents,
            **flow_loss_kwargs(args),
        )
        loss = (
            srr_loss
            + args.clean_loss_weight * clean_loss
            + args.endpoint_loss_weight * endpoint_loss
        )
        accelerator.print(json.dumps({
            "dry_run": True,
            "srr_loss": float(srr_loss.detach().cpu()),
            "clean_loss": float(clean_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
            "rollout_depth": actual_depth,
            "boundary_blend": args.boundary_blend_max,
            "self_forcing": args.self_forcing,
            "endpoint_loss": float(endpoint_loss.detach().cpu()),
        }))
        return

    global_step = int(args.start_step)
    epoch = 0
    model.train()
    while global_step < args.max_steps:
        dataset.set_epoch(epoch)
        for batch in loader:
            if global_step >= args.max_steps:
                break
            gt, actions, proprio = encode_batch(batch, vae, accelerator)
            rollout_depth, blend_ratio = scheduled_rollout(args, global_step)
            with accelerator.accumulate(model):
                if args.teacher_forced_random_position:
                    max_position = gt.shape[2] - args.history_latents - args.rollout_chunk_latents
                    position = int(torch.randint(max_position + 1, (1,), device=gt.device).item())
                    target_start = args.history_latents + position
                    history = gt[:, :, target_start - args.history_latents : target_start]
                    target = gt[:, :, target_start : target_start + args.rollout_chunk_latents]
                    window = torch.cat([history, target], dim=2)
                    action_window = actions[
                        :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                    ]
                    proprio_window = proprio[
                        :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                    ]
                    position_loss = continuation_flow_loss(
                        model,
                        window,
                        action_window,
                        proprio_window,
                        args.history_latents,
                        **flow_loss_kwargs(args),
                    )
                    accelerator.backward(position_loss)
                    srr_loss_value = float(position_loss.detach().cpu())
                    clean_loss_value = 0.0
                    if args.clean_loss_weight:
                        clean_window = gt[
                            :, :, : args.history_latents + args.rollout_chunk_latents
                        ]
                        clean_loss = continuation_flow_loss(
                            model,
                            clean_window,
                            actions[:, : args.history_latents + args.rollout_chunk_latents],
                            proprio[:, : args.history_latents + args.rollout_chunk_latents],
                            args.history_latents,
                            **flow_loss_kwargs(args),
                        )
                        accelerator.backward(args.clean_loss_weight * clean_loss)
                        clean_loss_value = float(clean_loss.detach().cpu())
                    total_loss_value = srr_loss_value + args.clean_loss_weight * clean_loss_value
                    rollout_values = [srr_loss_value]
                    endpoint_loss = torch.zeros((), device=gt.device, dtype=gt.dtype)
                    endpoint_consistency = torch.zeros((), device=gt.device, dtype=gt.dtype)
                elif args.self_forcing_random_position:
                    max_position = min(
                        int(args.rollout_depth),
                        (gt.shape[2] - args.history_latents) // args.rollout_chunk_latents,
                    ) - 1
                    min_position = min(
                        int(args.self_forcing_random_min_position), max_position
                    )
                    if max_position < 0:
                        raise RuntimeError("clip is too short for self-forcing random position")
                    position = int(torch.randint(
                        min_position, max_position + 1, (1,), device=gt.device
                    ).item())
                    entries = self_forcing_histories(
                        model,
                        gt,
                        actions,
                        proprio,
                        args,
                        rollout_depth=position + 1,
                        blend_ratio=blend_ratio,
                    )
                    history, target_start = entries[-1]
                    target = gt[
                        :, :, target_start : target_start + args.rollout_chunk_latents
                    ]
                    window = torch.cat([history, target], dim=2)
                    action_window = actions[
                        :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                    ]
                    proprio_window = proprio[
                        :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                    ]
                    position_loss = continuation_flow_loss(
                        model,
                        window,
                        action_window,
                        proprio_window,
                        args.history_latents,
                        **flow_loss_kwargs(args),
                    )
                    accelerator.backward(position_loss)
                    srr_loss_value = float(position_loss.detach().cpu())
                    clean_loss_value = 0.0
                    endpoint_loss = torch.zeros((), device=gt.device, dtype=gt.dtype)
                    endpoint_consistency = torch.zeros((), device=gt.device, dtype=gt.dtype)
                    if args.endpoint_loss_weight > 0 or args.endpoint_consistency_weight > 0:
                        endpoint_loss, endpoint_consistency = ensemble_endpoint_losses(
                            model, history, action_window, proprio_window, target, args
                        )
                        auxiliary_loss = (
                            args.endpoint_loss_weight * endpoint_loss
                            + args.endpoint_consistency_weight * endpoint_consistency
                        )
                        if auxiliary_loss.requires_grad:
                            accelerator.backward(auxiliary_loss)
                        del auxiliary_loss
                    if args.clean_loss_weight:
                        clean_window = gt[
                            :, :, : args.history_latents + args.rollout_chunk_latents
                        ]
                        clean_loss = continuation_flow_loss(
                            model,
                            clean_window,
                            actions[:, : args.history_latents + args.rollout_chunk_latents],
                            proprio[:, : args.history_latents + args.rollout_chunk_latents],
                            args.history_latents,
                            **flow_loss_kwargs(args),
                        )
                        accelerator.backward(args.clean_loss_weight * clean_loss)
                        clean_loss_value = float(clean_loss.detach().cpu())
                    total_loss_value = srr_loss_value + args.clean_loss_weight * clean_loss_value
                    total_loss_value += args.endpoint_loss_weight * float(endpoint_loss.detach().cpu())
                    total_loss_value += args.endpoint_consistency_weight * float(endpoint_consistency.detach().cpu())
                    rollout_values = [srr_loss_value]
                elif args.self_forcing:
                    entries = self_forcing_histories(
                        model,
                        gt,
                        actions,
                        proprio,
                        args,
                        rollout_depth=rollout_depth,
                        blend_ratio=blend_ratio,
                    )
                    weights = [
                        args.rollout_loss_decay ** i
                        * args.rollout_position_growth ** i
                        for i in range(len(entries))
                    ]
                    weight_sum = float(sum(weights))
                    rollout_values = []
                    selected_entries = (
                        entries
                        if args.self_forcing_position < 0
                        else [entries[args.self_forcing_position]]
                    )
                    selected_weights = (
                        weights if args.self_forcing_position < 0 else [1.0]
                    )
                    selected_weight_sum = float(sum(selected_weights))
                    for (history, target_start), weight in zip(selected_entries, selected_weights):
                        target = gt[
                            :, :, target_start : target_start + args.rollout_chunk_latents
                        ]
                        window = torch.cat([history, target], dim=2)
                        action_window = actions[
                            :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                        ]
                        proprio_window = proprio[
                            :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                        ]
                        position_loss = continuation_flow_loss(
                            model,
                            window,
                            action_window,
                            proprio_window,
                            args.history_latents,
                            **flow_loss_kwargs(args),
                        )
                        rollout_values.append(float(position_loss.detach().cpu()))
                        accelerator.backward(position_loss * (weight / selected_weight_sum))

                    clean_window = gt[
                        :, :, : args.history_latents + args.rollout_chunk_latents
                    ]
                    clean_loss = continuation_flow_loss(
                        model,
                        clean_window,
                        actions[:, : args.history_latents + args.rollout_chunk_latents],
                        proprio[:, : args.history_latents + args.rollout_chunk_latents],
                        args.history_latents,
                        **flow_loss_kwargs(args),
                    )
                    clean_loss_value = float(clean_loss.detach().cpu())
                    if args.clean_loss_weight:
                        accelerator.backward(args.clean_loss_weight * clean_loss)
                    endpoint_loss = torch.zeros((), device=gt.device, dtype=gt.dtype)
                    endpoint_consistency = torch.zeros((), device=gt.device, dtype=gt.dtype)
                    if args.endpoint_loss_weight > 0:
                        endpoint_indices = [0] if args.endpoint_loss_positions == "first" else [-1]
                        if args.endpoint_loss_positions == "both":
                            endpoint_indices = [0, -1]
                        if args.endpoint_loss_positions == "all":
                            endpoint_indices = list(range(len(entries)))
                        endpoint_values = []
                        endpoint_consistency_values = []
                        endpoint_count = max(1, len(endpoint_indices))
                        for entry_index in endpoint_indices:
                            history, target_start = entries[entry_index]
                            target = gt[
                                :, :, target_start : target_start + args.rollout_chunk_latents
                            ]
                            action_window = actions[
                                :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                            ]
                            proprio_window = proprio[
                                :, target_start - args.history_latents : target_start + args.rollout_chunk_latents
                            ]
                            endpoint_value, consistency_value = ensemble_endpoint_losses(
                                model, history, action_window, proprio_window, target, args
                            )
                            endpoint_values.append(float(endpoint_value.detach().cpu()))
                            endpoint_consistency_values.append(float(consistency_value.detach().cpu()))
                            # Backpropagate one position at a time.  Keeping
                            # every multi-step Euler graph alive until the end
                            # exceeds the 40GB device limit for endpoint=all.
                            auxiliary_loss = (
                                args.endpoint_loss_weight * endpoint_value
                                + args.endpoint_consistency_weight * consistency_value
                            ) / endpoint_count
                            if auxiliary_loss.requires_grad:
                                accelerator.backward(auxiliary_loss)
                            del endpoint_value, consistency_value, auxiliary_loss
                        endpoint_loss = torch.tensor(
                            sum(endpoint_values) / endpoint_count,
                            device=gt.device,
                            dtype=gt.dtype,
                        )
                        endpoint_consistency = torch.tensor(
                            sum(endpoint_consistency_values) / endpoint_count,
                            device=gt.device,
                            dtype=gt.dtype,
                        )
                    srr_loss_value = float(
                        sum(value * weight for value, weight in zip(rollout_values, selected_weights))
                        / selected_weight_sum
                    )
                    total_loss_value = srr_loss_value + args.clean_loss_weight * clean_loss_value
                    total_loss_value += args.endpoint_loss_weight * float(endpoint_loss.detach().cpu())
                    total_loss_value += args.endpoint_consistency_weight * float(endpoint_consistency.detach().cpu())
                else:
                    history, target_start = polluted_history(
                        model,
                        gt,
                        actions,
                        proprio,
                        args,
                        rollout_depth=rollout_depth,
                        blend_ratio=blend_ratio,
                    )
                    target = gt[:, :, target_start : target_start + args.rollout_chunk_latents]
                    window = torch.cat([history, target], dim=2)
                    action_window = actions[:, target_start - args.history_latents : target_start + args.rollout_chunk_latents]
                    proprio_window = proprio[:, target_start - args.history_latents : target_start + args.rollout_chunk_latents]
                    srr_loss = continuation_flow_loss(
                        model, window, action_window, proprio_window, args.history_latents
                        , **flow_loss_kwargs(args)
                    )
                    clean_window = gt[:, :, : args.history_latents + args.rollout_chunk_latents]
                    clean_loss = continuation_flow_loss(
                        model,
                        clean_window,
                        actions[:, : args.history_latents + args.rollout_chunk_latents],
                        proprio[:, : args.history_latents + args.rollout_chunk_latents],
                        args.history_latents,
                        **flow_loss_kwargs(args),
                    )
                    loss = srr_loss + args.clean_loss_weight * clean_loss
                    accelerator.backward(loss)
                    srr_loss_value = float(srr_loss.detach().cpu())
                    clean_loss_value = float(clean_loss.detach().cpu())
                    total_loss_value = float(loss.detach().cpu())

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "srr_loss": srr_loss_value,
                                "clean_loss": clean_loss_value,
                                "total_loss": total_loss_value,
                                "rollout_depth": (
                                    len(entries)
                                    if (args.self_forcing or args.self_forcing_random_position)
                                    else 1
                                ),
                                "boundary_blend": float(blend_ratio),
                                "self_forcing": args.self_forcing or args.self_forcing_random_position,
                                "self_forcing_random_position": args.self_forcing_random_position,
                                "teacher_forced_random_position": args.teacher_forced_random_position,
                                "first_position_loss": rollout_values[0] if args.self_forcing else srr_loss_value,
                                "last_position_loss": rollout_values[-1] if args.self_forcing else srr_loss_value,
                                "endpoint_loss": (
                                    float(endpoint_loss.detach().cpu())
                                    if (args.self_forcing or args.self_forcing_random_position)
                                    else 0.0
                                ),
                                "target_position": int(target_start) if (args.teacher_forced_random_position or args.self_forcing_random_position) else None,
                            }
                        ),
                        flush=True,
                    )
                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, args.output_dir, global_step)
        epoch += 1
    save_checkpoint(accelerator, args.output_dir, global_step)


if __name__ == "__main__":
    main()
