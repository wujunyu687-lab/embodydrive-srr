"""Dataset for the offline wrist MP4 + Wan-latent cache."""

import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class DroidWristLatentDataset(Dataset):
    """Sample 17 processed 5Hz frames around a 5-frame Wan latent window."""

    def __init__(
        self,
        data_root,
        frames=17,
        split="train",
        max_episodes=0,
        seed=42,
        include_video=False,
        stats_path=None,
    ):
        self.data_root = Path(data_root)
        self.frames = int(frames)
        self.split = split
        self.seed = int(seed)
        self.include_video = bool(include_video)
        self.epoch = 0
        self.latent_ratio = 4
        self.latent_window = (self.frames - 1) // self.latent_ratio + 1
        self.annotations = []
        candidate_paths = sorted(self.data_root.glob(f"annotation/{split}/*.json"))
        wanted_episodes = int(max_episodes) if max_episodes > 0 else None
        for path in candidate_paths:
            try:
                annotation = json.loads(path.read_text())
                latent_path = annotation.get("latent_path")
                if latent_path is None:
                    latent_path = (
                        Path("latent_videos")
                        / annotation["split"]
                        / annotation["episode_id"]
                        / "wrist.pt"
                    )
                # A fixed 17-frame training window is required.  The source
                # dataset contains a small number of very short episodes;
                # keep them out of the index instead of failing inside
                # __getitem__ after DDP has already started.
                if (
                    int(annotation.get("video_length", 0)) >= self.frames
                    and (self.data_root / latent_path).exists()
                ):
                    self.annotations.append(path)
                    if wanted_episodes is not None and len(self.annotations) >= wanted_episodes:
                        break
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        if not self.annotations:
            raise FileNotFoundError(
                f"No offline annotations found under {self.data_root}/annotation/{split}"
            )
        self.stats = None
        if stats_path is not None and Path(stats_path).exists():
            self.stats = json.loads(Path(stats_path).read_text())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.annotations)

    def _normalize(self, values, kind):
        if self.stats is None:
            return values
        p01 = torch.tensor(self.stats[kind]["p01"], dtype=values.dtype)
        p99 = torch.tensor(self.stats[kind]["p99"], dtype=values.dtype)
        return ((2.0 * (values - p01) / (p99 - p01 + 1e-8)) - 1.0).clamp(-1.0, 1.0)

    def _latent_conditions(self, values, latent_start, latent_count, kind):
        """Aggregate raw conditions using the full-video Wan latent clock.

        Wan's causal VAE has one single-frame latent followed by four-frame
        groups.  Latents are cached from the complete episode, so conditions
        must use the same global grouping even when the training window starts
        in the middle of an episode.
        """
        # Aggregate rotations in raw radians before robust normalization.
        # Normalizing first moves the +/-pi branch cut and makes circular
        # averaging incorrect.
        values = torch.tensor(values, dtype=torch.float32)
        latent_total = 1 + max(0, (values.shape[0] - 1) // self.latent_ratio)
        usable = 1 + max(0, latent_total - 1) * self.latent_ratio
        if values.shape[0] < usable:
            values = torch.cat(
                [values, values[-1:].repeat(usable - values.shape[0], 1)], dim=0
            )
        values = values[:usable]
        grouped = [values[:1]]
        if latent_total > 1:
            grouped_values = values[1:].reshape(
                latent_total - 1, self.latent_ratio, -1
            )
            grouped_mean = grouped_values.mean(dim=1)
            circular_dims = (3, 4, 5) if kind == "action" else (10, 11, 12)
            dims = torch.as_tensor(circular_dims, dtype=torch.long)
            angles = grouped_values.index_select(-1, dims)
            circular = torch.atan2(
                torch.sin(angles).mean(dim=1), torch.cos(angles).mean(dim=1)
            )
            grouped_mean = grouped_mean.clone()
            grouped_mean.index_copy_(-1, dims, circular)
            grouped.append(grouped_mean)
        grouped = torch.cat(grouped, dim=0)
        end = latent_start + latent_count
        if end > grouped.shape[0]:
            grouped = torch.cat(
                [grouped, grouped[-1:].repeat(end - grouped.shape[0], 1)], dim=0
            )
        return self._normalize(grouped[latent_start:end], kind)

    def __getitem__(self, index):
        annotation = json.loads(self.annotations[index].read_text())
        latent_path = annotation.get("latent_path")
        if latent_path is None:
            latent_path = (
                Path("latent_videos")
                / annotation["split"]
                / annotation["episode_id"]
                / "wrist.pt"
            )
        latent = torch.load(self.data_root / latent_path, map_location="cpu")
        latent = latent.contiguous()
        video_length = int(annotation["video_length"])
        max_start = min(
            latent.shape[0] - self.latent_window,
            (video_length - self.frames) // self.latent_ratio,
        )
        if max_start < 0:
            raise RuntimeError(
                f"Episode {annotation['episode_id']} is too short: "
                f"video={video_length}, latent={latent.shape[0]}, frames={self.frames}"
            )
        rng = random.Random(self.seed + self.epoch * 100003 + index)
        latent_start = rng.randint(0, max_start) if max_start > 0 else 0
        frame_start = latent_start * self.latent_ratio
        frame_end = frame_start + self.frames
        sample = {
            "latents": latent[latent_start : latent_start + self.latent_window],
            "action": self._latent_conditions(
                annotation["action"], latent_start, self.latent_window, "action"
            ),
            "proprio": self._latent_conditions(
                annotation["proprio"], latent_start, self.latent_window, "proprio"
            ),
        }
        if self.include_video:
            video_path = self.data_root / annotation["video_path"]
            reader = imageio.get_reader(str(video_path))
            frames = np.stack(
                [reader.get_data(i) for i in range(frame_start, frame_end)], axis=0
            )
            reader.close()
            sample["video"] = torch.from_numpy(frames).float() / 127.5 - 1.0
        return sample
