"""Small streaming reader for the RLDS-style DROID TFRecord shards."""

from io import BytesIO
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from tfrecord.reader import tfrecord_loader


class DroidVideoActionDataset(IterableDataset):
    """Yield short external-camera clips and aligned 7D actions.

    This is intentionally a smoke/G0 reader. It skips incomplete ``.part``
    files and does not build a global episode index, which keeps startup cheap.
    """

    def __init__(
        self,
        data_root,
        frames=17,
        image_size=64,
        max_shards=0,
        max_episodes=0,
        seed=42,
        random_crop=True,
        include_proprio=True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.frames = int(frames)
        self.image_size = int(image_size)
        self.max_shards = int(max_shards)
        self.max_episodes = int(max_episodes)
        self.seed = int(seed)
        self.random_crop = bool(random_crop)
        self.include_proprio = bool(include_proprio)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _shards(self):
        paths = sorted(
            p for p in self.data_root.glob("droid_101-train.tfrecord-*-of-02048")
            if not p.name.endswith(".part")
        )
        if not paths:
            raise FileNotFoundError(f"No complete DROID shards found under {self.data_root}")
        return paths[: self.max_shards] if self.max_shards > 0 else paths

    def _decode_frames(self, encoded_frames):
        output = []
        for encoded in encoded_frames:
            image = Image.open(BytesIO(bytes(encoded).rstrip(b"\x00"))).convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            output.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(output)

    @staticmethod
    def _reshape_steps(record, key, num_frames, dim):
        values = np.asarray(record[key], dtype=np.float32)
        return values.reshape(num_frames, dim)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        yielded = 0
        rng = random.Random(self.seed + self.epoch * 100003 + worker_id)
        for shard_index, shard in enumerate(self._shards()):
            if shard_index % worker_count != worker_id:
                continue
            for record in tfrecord_loader(str(shard), None):
                if self.max_episodes > 0 and yielded >= self.max_episodes:
                    return
                image_bytes = record["steps/observation/exterior_image_1_left"]
                num_frames = len(image_bytes)
                if num_frames < self.frames:
                    continue
                try:
                    actions = self._reshape_steps(record, "steps/action", num_frames, 7)
                except (KeyError, ValueError):
                    # Some DROID records contain an image sequence but only a
                    # single action vector. They cannot be aligned safely and
                    # must be skipped consistently by every DDP rank.
                    continue
                if self.random_crop:
                    start = rng.randint(0, num_frames - self.frames)
                else:
                    start = 0
                end = start + self.frames
                clip = self._decode_frames(image_bytes[start:end])
                action_clip = torch.from_numpy(actions[start:end].copy())
                sample = {
                    "video": clip,
                    "action": action_clip,
                }
                if self.include_proprio:
                    try:
                        joint = self._reshape_steps(
                            record, "steps/observation/joint_position", num_frames, 7
                        )
                        cartesian = self._reshape_steps(
                            record, "steps/observation/cartesian_position", num_frames, 6
                        )
                        gripper = self._reshape_steps(
                            record, "steps/observation/gripper_position", num_frames, 1
                        )
                    except (KeyError, ValueError):
                        continue
                    proprio = np.concatenate([joint, cartesian, gripper], axis=-1)
                    proprio = np.nan_to_num(proprio, nan=0.0, posinf=0.0, neginf=0.0)
                    sample["proprio"] = torch.from_numpy(proprio[start:end].copy())
                yield sample
                yielded += 1
