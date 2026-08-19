"""正式 G0 训练入口：DROID 视频 + action/proprio 条件的 Wan flow matching。"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
import imageio.v2 as imageio
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embodydrive.droid_dataset import DroidVideoActionDataset
from embodydrive.offline_dataset import DroidWristLatentDataset
from embodydrive.rollout import euler_sample_chunk
from videox_fun.models.wan_vae import AutoencoderKLWan
from videox_fun.models.wan_transformer3d_unified_6v.model import UnifiedTransformer3DModel


def parse_args():
    parser = argparse.ArgumentParser(description="EmbodyDrive G0 training")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--offline-root", default=None, help="Use offline wrist MP4/latent cache")
    parser.add_argument("--stats-path", default=None, help="Robust action/proprio stats JSON")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--history-latents", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all complete shards")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all episodes per epoch")
    parser.add_argument("--max-val-episodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--visualization-steps", type=int, default=500)
    parser.add_argument("--visualization-videos", type=int, default=2)
    parser.add_argument("--visualization-rollout-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load data/model and run one forward loss only")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--no-proprio", action="store_true")
    return parser.parse_args()


def downsample_temporal(values, latent_frames, temporal_ratio=4, circular_dims=()):
    """Align raw-frame conditions to Wan's causal VAE latent frames."""
    if values.shape[1] == latent_frames:
        return values
    required = 1 + max(0, latent_frames - 1) * temporal_ratio
    if values.shape[1] < required:
        values = torch.cat(
            [values, values[:, -1:].repeat(1, required - values.shape[1], 1)], dim=1
        )
    first = values[:, :1]
    if latent_frames == 1:
        return first
    grouped = values[:, 1:required].reshape(
        values.shape[0], latent_frames - 1, temporal_ratio, values.shape[-1]
    )
    rest = grouped.mean(dim=2)
    if circular_dims:
        dims = torch.as_tensor(tuple(circular_dims), device=values.device)
        # `values` are expected in raw units here; normalize after this
        # aggregation when callers use circular pose channels.
        angles = grouped.index_select(-1, dims)
        circular = torch.atan2(
            torch.sin(angles).mean(dim=2), torch.cos(angles).mean(dim=2)
        )
        rest = rest.clone()
        rest.index_copy_(-1, dims, circular)
    return torch.cat([first, rest], dim=1)


def build_transformer(model_root, use_action, use_proprio, dtype):
    additional = {
        "dict_mapping": {"in_dim": "in_channels", "dim": "hidden_size"},
        "use_view_embedding": False,
        "position_embedding_kwargs": {
            "type": "videox_fun.models.wan_transformer3d_unified_6v.RopeEmb"
        },
        "additional_condition_kwargs": {},
    }
    if use_action:
        additional["additional_condition_kwargs"]["action"] = {
            "type": "embodydrive.robot_conditioning.RobotActionConditioningProj",
            "kwargs": {"in_dim": 7},
        }
    if use_proprio:
        additional["additional_condition_kwargs"]["proprio"] = {
            "type": "embodydrive.robot_conditioning.RobotProprioConditioningProj",
            "kwargs": {"in_dim": 14},
        }
    additional["additional_condition_kwargs"]["history"] = {
        "type": "embodydrive.robot_conditioning.RobotHistoryConditioningProj",
        "kwargs": {"in_dim": 17},
    }
    if not additional["additional_condition_kwargs"]:
        additional.pop("additional_condition_kwargs")
    return UnifiedTransformer3DModel.from_pretrained(
        model_root,
        transformer_additional_kwargs=additional,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    )


def seq_len_for_latents(model, latents):
    patch_size = tuple(model.config.patch_size)
    return math.ceil(
        latents.shape[2] * latents.shape[3] * latents.shape[4]
        / math.prod(patch_size)
    )


def make_history_condition(latents, history_frames):
    if history_frames <= 0 or history_frames >= latents.shape[2]:
        raise ValueError(
            f"history-latents must be in [1, {latents.shape[2] - 1}], got {history_frames}"
        )
    history = torch.zeros(
        latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4],
        device=latents.device, dtype=latents.dtype
    )
    history[:, :, :history_frames] = 1
    return torch.cat(
        [
            torch.cat(
                [latents[:, :, :history_frames], torch.zeros_like(latents[:, :, history_frames:])],
                dim=2,
            ),
            history,
        ],
        dim=1,
    )


def make_conditions(batch, latents, history_frames, use_action, use_proprio):
    conditions = {"history": make_history_condition(latents, history_frames)}
    latent_frames = latents.shape[2]
    if use_action:
        conditions["action"] = downsample_temporal(
            batch["action"], latent_frames, circular_dims=(3, 4, 5)
        )
    if use_proprio:
        conditions["proprio"] = downsample_temporal(
            batch["proprio"], latent_frames, circular_dims=(10, 11, 12)
        )
    return conditions


def flow_matching_loss(model, vae, batch, accelerator, history_frames, use_action, use_proprio):
    device = accelerator.device
    dtype = torch.bfloat16
    if "latents" in batch:
        latents = batch["latents"].to(device, dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()
        batch_size = latents.shape[0]
    else:
        video = batch["video"].to(device, dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            with accelerator.autocast():
                latents = vae.encode(video).latent_dist.sample()
        batch_size = video.shape[0]
    condition_batch = {"action": batch["action"].to(device, dtype=dtype)}
    if use_proprio:
        condition_batch["proprio"] = batch["proprio"].to(device, dtype=dtype)
    conditions = make_conditions(condition_batch, latents, history_frames, use_action, use_proprio)
    sigma = torch.rand(batch_size, device=device, dtype=dtype)
    sigma_view = sigma.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(latents)
    noisy_future = (1.0 - sigma_view) * latents + sigma_view * noise
    noisy = torch.cat(
        [latents[:, :, :history_frames], noisy_future[:, :, history_frames:]], dim=2
    )
    context = [torch.zeros(512, 4096, device=device, dtype=dtype) for _ in range(batch_size)]
    with accelerator.autocast():
        prediction = model(
            noisy,
            t=sigma,
            context=context,
            seq_len=seq_len_for_latents(accelerator.unwrap_model(model), latents),
            num_views=1,
            dtype=dtype,
            crossview_attn_type="full",
            additional_conditions=conditions,
        )
        target = noise - latents
        future_mask = torch.zeros_like(target)
        future_mask[:, :, history_frames:] = 1
        error = ((prediction - target).float() * future_mask.float()).square()
        loss = error.sum() / future_mask.float().sum().clamp_min(1.0)
    return loss


@torch.no_grad()
def evaluate(model, vae, loader, accelerator, history_frames, use_action, use_proprio, limit):
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        losses.append(flow_matching_loss(model, vae, batch, accelerator, history_frames, use_action, use_proprio))
    model.train()
    if not losses:
        return float("nan")
    value = torch.stack(losses).mean()
    return float(accelerator.gather_for_metrics(value.detach().reshape(1)).mean().cpu())


@torch.no_grad()
def render_validation_videos(
    model,
    vae,
    loader,
    accelerator,
    args,
    use_action,
    use_proprio,
    step,
):
    """Save prediction, ground truth, and side-by-side GIFs on fixed clips."""
    output_dir = Path(args.output_dir) / "visualizations" / f"step-{step:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = accelerator.unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    rendered = 0
    for batch in loader:
        if rendered >= args.visualization_videos:
            break
        raw_video = batch.get("video")
        if "latents" in batch:
            latents = batch["latents"].to(accelerator.device, dtype=torch.bfloat16)
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            video = raw_video.to(accelerator.device, dtype=torch.bfloat16)
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            with accelerator.autocast():
                latents = vae.encode(video).latent_dist.sample()
        history_frames = args.history_latents
        if history_frames >= latents.shape[2]:
            continue
        history = latents[:, :, :history_frames]
        action = None
        proprio = None
        if use_action:
            action = downsample_temporal(
                batch["action"].to(accelerator.device, dtype=torch.bfloat16), latents.shape[2]
            )
        if use_proprio:
            proprio = downsample_temporal(
                batch["proprio"].to(accelerator.device, dtype=torch.bfloat16), latents.shape[2]
            )
        future_frames = latents.shape[2] - history_frames
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.seed + step + rendered
        )
        noise = torch.randn(
            latents.shape[0],
            latents.shape[1],
            future_frames,
            latents.shape[3],
            latents.shape[4],
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        with accelerator.autocast():
            predicted_future = euler_sample_chunk(
                base_model,
                history,
                action,
                proprio,
                future_frames,
                args.visualization_rollout_steps,
                noise=noise,
            )
            predicted_latents = torch.cat([history, predicted_future], dim=2)
            decoded = vae.decode(torch.cat([predicted_latents, latents], dim=0)).sample
        decoded = ((decoded.float().clamp(-1, 1) + 1.0) * 127.5).byte()
        batch_size = latents.shape[0]
        predicted = decoded[:batch_size].permute(0, 2, 3, 4, 1).cpu().numpy()
        if raw_video is not None:
            ground_truth = (
                ((raw_video.float().clamp(-1, 1) + 1.0) * 127.5)
                .byte()
                .cpu()
                .numpy()
            )
        else:
            ground_truth = decoded[batch_size:].permute(0, 2, 3, 4, 1).cpu().numpy()
        for sample_index in range(batch_size):
            pred_frames = predicted[sample_index]
            gt_frames = ground_truth[sample_index]
            compare_frames = np.concatenate([pred_frames, gt_frames], axis=2)
            stem = output_dir / f"video-{rendered:02d}"
            imageio.mimsave(f"{stem}-pred.mp4", pred_frames, fps=10, codec="libx264", macro_block_size=1)
            imageio.mimsave(f"{stem}-gt.mp4", gt_frames, fps=10, codec="libx264", macro_block_size=1)
            imageio.mimsave(f"{stem}-compare.mp4", compare_frames, fps=10, codec="libx264", macro_block_size=1)
            rendered += 1
    if was_training:
        base_model.train()
    return str(output_dir) if rendered else None


def save_checkpoint(accelerator, output_dir, step):
    accelerator.wait_for_everyone()
    path = Path(output_dir) / f"checkpoint-{step:08d}"
    accelerator.save_state(str(path))
    if accelerator.is_main_process:
        print(f"saved checkpoint: {path}", flush=True)


def log_metrics(accelerator, output_dir, values):
    """Print and persist one JSON metrics record on the main process."""
    if accelerator.is_main_process:
        line = json.dumps(values, sort_keys=True)
        with open(Path(output_dir) / "metrics.jsonl", "a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def main():
    args = parse_args()
    if not args.offline_root and not args.data_root:
        raise ValueError("provide --offline-root or --data-root")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for G0 training")
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    set_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_dir=args.output_dir,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device
    dtype = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        with open(Path(args.output_dir) / "training_args.json", "w") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)

    use_action = not args.no_action
    use_proprio = not args.no_proprio
    if args.offline_root:
        stats_path = args.stats_path or str(Path(args.offline_root) / "stats.json")
        train_dataset = DroidWristLatentDataset(
            args.offline_root,
            frames=args.frames,
            split="train",
            max_episodes=args.max_episodes,
            seed=args.seed,
            include_video=False,
            stats_path=stats_path,
        )
        val_dataset = DroidWristLatentDataset(
            args.offline_root,
            frames=args.frames,
            split="val",
            max_episodes=args.max_val_episodes,
            seed=args.seed + 100000,
            include_video=True,
            stats_path=stats_path,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=0)
    else:
        train_dataset = DroidVideoActionDataset(
            args.data_root,
            frames=args.frames,
            image_size=args.image_size,
            max_shards=args.max_shards,
            max_episodes=args.max_episodes,
            seed=args.seed,
            random_crop=True,
            include_proprio=use_proprio,
        )
        val_dataset = DroidVideoActionDataset(
            args.data_root,
            frames=args.frames,
            image_size=args.image_size,
            max_shards=max(1, min(args.max_shards or 1, 1)),
            max_episodes=args.max_val_episodes,
            seed=args.seed + 100000,
            random_crop=False,
            include_proprio=use_proprio,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=0)

    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path,
        additional_kwargs={"temporal_compression_ratio": 4, "spatial_compression_ratio": 8},
    ).to(device, dtype=dtype).eval()
    vae.requires_grad_(False)
    model = build_transformer(args.model_root, use_action, use_proprio, dtype)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    global_step = 0
    if args.resume:
        accelerator.load_state(args.resume)
        match = re.search(r"checkpoint-(\d+)$", str(args.resume).rstrip("/"))
        if match:
            global_step = int(match.group(1))
        accelerator.print(f"resumed from {args.resume}")

    if args.dry_run:
        batch = next(iter(train_loader))
        loss = flow_matching_loss(model, vae, batch, accelerator, args.history_latents, use_action, use_proprio)
        accelerator.print(json.dumps({"dry_run": True, "loss": float(loss.detach().cpu())}))
        return

    epoch = 0
    model.train()
    while global_step < args.max_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if global_step >= args.max_steps:
                break
            with accelerator.accumulate(model):
                loss = flow_matching_loss(model, vae, batch, accelerator, args.history_latents, use_action, use_proprio)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    log_metrics(
                        accelerator,
                        args.output_dir,
                        {"step": global_step, "loss": float(loss.detach().cpu())},
                    )
                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    val_loss = evaluate(model, vae, val_loader, accelerator, args.history_latents, use_action, use_proprio, args.max_val_episodes)
                    visualization_dir = None
                    if args.visualization_steps > 0 and global_step % args.visualization_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            try:
                                visualization_dir = render_validation_videos(
                                    model,
                                    vae,
                                    val_loader,
                                    accelerator,
                                    args,
                                    use_action,
                                    use_proprio,
                                    global_step,
                                )
                            except Exception as exc:
                                # A visualization failure must not kill the
                                # distributed training job after validation.
                                print(
                                    f"visualization failed at step {global_step}: {exc!r}",
                                    flush=True,
                                )
                        accelerator.wait_for_everyone()
                    log_metrics(
                        accelerator,
                        args.output_dir,
                        {
                            "step": global_step,
                            "val_loss": val_loss,
                            "visualization_dir": visualization_dir,
                        },
                    )
                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, args.output_dir, global_step)
        epoch += 1

    save_checkpoint(accelerator, args.output_dir, global_step)
    accelerator.print(f"finished G0 training at step {global_step}")


if __name__ == "__main__":
    main()
