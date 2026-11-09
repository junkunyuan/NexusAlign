# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Portions adapted from finite-difference-flow-optimization/fdfo/prompts.py.
# Licensed under the NVIDIA Source Code License v1 (Non-Commercial).
# See licenses/FDFO-LICENSE.txt for the full license.

"""Local FDFO prompts and rank-local continuous prompt streams."""

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


@dataclass
class PromptBatch:
    prompts: list[str]
    metadata: list[dict] | None = None
    prompt_ids: list[int] | None = None
    canonical_prompt_ids: list[int] | None = None


class TwinTiltPromptDataset(Dataset):
    """Keep source row order and duplicate prompts, with isolated metadata copies."""

    def __init__(self, cfg_data, split: str | None = None) -> None:
        self.name = cfg_data.name
        self.split = split or cfg_data.split
        self.file_path = Path(cfg_data.path) / cfg_data.file.format(split=self.split)
        self.records = self._read_records()
        if not self.records:
            raise ValueError(f"No prompt rows found in {self.file_path}")
        self._attach_prompt_identity()
        self.texts = [record["prompt"] for record in self.records]
        self.metadatas = [copy.deepcopy(row["metadata"] or {}) for row in self.records]
        self.has_metadata = any(row["metadata"] is not None for row in self.records)
        print(
            f"📚 Dataset: {self.name}, split: {self.split}, "
            f"sample_count: {len(self)}"
        )

    def _read_records(self) -> list[dict]:
        records = []
        with self.file_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                text = line.strip()
                if not text:
                    continue
                if self.file_path.suffix != ".jsonl":
                    records.append({"prompt": text, "metadata": None})
                    continue
                if text.startswith("#"):
                    continue
                location = f"{self.file_path}:{line_number}"
                try:
                    metadata = json.loads(text)
                except json.JSONDecodeError as error:
                    message = f"{location}: invalid JSON: {error.msg}"
                    raise ValueError(message) from error
                if not isinstance(metadata, dict):
                    raise ValueError(f"{location} must be a JSON object")
                prompt = str(metadata.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError(f"{location} must contain a prompt")
                metadata["prompt"] = prompt
                records.append({"prompt": prompt, "metadata": metadata})
        return records

    def _attach_prompt_identity(self) -> None:
        canonical_ids = {}
        digest = hashlib.sha256()
        for prompt_id, record in enumerate(self.records):
            payload = record["metadata"]
            if payload is None:
                payload = {"prompt": record["prompt"]}
            canonical = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            canonical_id = canonical_ids.setdefault(canonical, len(canonical_ids))
            record["prompt_id"] = prompt_id
            record["canonical_prompt_id"] = canonical_id
            digest.update(
                json.dumps(
                    {"prompt": record["prompt"], "metadata": record["metadata"]},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        self.dataset_fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "text": record["prompt"],
            "metadata": copy.deepcopy(record["metadata"] or {}),
            "prompt_id": record["prompt_id"],
            "canonical_prompt_id": record["canonical_prompt_id"],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> tuple[list[str], list[dict]]:
        return [row["text"] for row in batch], [row["metadata"] for row in batch]

    def prompt_batch(self, indices: list[int]) -> PromptBatch:
        rows = [self[index] for index in indices]
        return PromptBatch(
            prompts=[row["text"] for row in rows],
            metadata=[row["metadata"] for row in rows] if self.has_metadata else None,
            prompt_ids=[row["prompt_id"] for row in rows],
            canonical_prompt_ids=[row["canonical_prompt_id"] for row in rows],
        )


@dataclass(frozen=True)
class PromptBatchPlan:
    """Even group batches across the actual training ranks."""

    world_size: int
    batch_size: int
    num_batches_per_epoch: int

    @classmethod
    def build(cls, total_groups_per_epoch=48, world_size=1,
              batch_size=None, max_batch_size=28):
        for name, value in (
            ("groups_per_epoch", total_groups_per_epoch),
            ("world_size", world_size), ("max_batch_size", max_batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if total_groups_per_epoch % world_size:
            raise ValueError(
                "groups_per_epoch must divide across the actual world size"
            )
        local = total_groups_per_epoch // world_size
        if batch_size is None:
            batch_size = next(size for size in range(min(local, max_batch_size), 0, -1)
                              if local % size == 0)
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size < 1 or local % batch_size):
            raise ValueError(
                "sample_batch_size must be a positive divisor of local groups"
            )
        return cls(world_size, batch_size, local // batch_size)


class PromptDataLoader:
    """Continuously reshuffle each rank's stream; persist its private RNG."""

    def __init__(self, dataset, rank, batch_plan, seed=6):
        if not 0 <= rank < batch_plan.world_size:
            raise ValueError("rank must be within the actual world size")
        self.dataset, self.rank, self.batch_plan = dataset, rank, batch_plan
        self.seed = seed
        self.rng = np.random.RandomState(seed + rank)
        self.indices = np.arange(len(dataset))
        self.rng.shuffle(self.indices)
        self.position = 0

    def get_prompt_batch(self, batch_size=None):
        count = self.batch_plan.batch_size if batch_size is None else batch_size
        if count != self.batch_plan.batch_size:
            raise ValueError("batch size must match the actual rank batch plan")
        indices = []
        for _ in range(count):
            if self.position == len(self.indices):
                self.rng.shuffle(self.indices)
                self.position = 0
            indices.append(int(self.indices[self.position]))
            self.position += 1
        return self.dataset.prompt_batch(indices)

    def state_dict(self):
        return {
            "seed": self.seed, "rank": self.rank,
            "world_size": self.batch_plan.world_size,
            "dataset_fingerprint": self.dataset.dataset_fingerprint,
            "indices": self.indices.copy(), "position": self.position,
            "rng": self.rng.get_state(),
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("seed", "rank", "world_size", "dataset_fingerprint"):
            if state[key] != expected[key]:
                raise ValueError(f"Prompt stream has a different {key}")
        indices = np.asarray(state["indices"])
        if not np.array_equal(np.sort(indices), np.arange(len(self.dataset))):
            raise ValueError("Prompt stream indices are not a dataset permutation")
        if (not isinstance(state["position"], int)
                or not 0 <= state["position"] <= len(indices)):
            raise ValueError("Invalid prompt stream position")
        self.indices = indices.copy()
        self.position = state["position"]
        self.rng.set_state(state["rng"])
