"""ImageNet-1K dataset: parquet reader and cached VAE-latent loader."""

import io
import os
import glob
import json
import time
import bisect
from typing import Any
from collections.abc import Callable

import numpy as np
import pyarrow as pa
from PIL import Image
import pyarrow.parquet as pq
from safetensors import safe_open

import torch
import torch.distributed as dist

from nexus_align.utils.progress import TqdmBar
from nexus_align.datasets.base_dataset import BaseTextImageDataset

NUM_CLASSES = 1000


class ImageNet1K(BaseTextImageDataset):
    """
    ImageNet-1K dataset with raw and latent modes.

    Source layout (huggingface imagenet-1k, ~156GB under ILSVRC/imagenet-1k/data):
    - Parquet shards named {split}-{i:05d}-of-{n:05d}.parquet, each ~500MB with
      5 row groups: train 294 shards (1281167 rows), validation 14 (50000),
      test 28 (100000).
    - Columns: image struct<bytes: binary, path: string> holding the raw JPEG
      bytes and the original filename, and label int64 in [0, 1000) (-1 for the
      unlabeled test split). Images vary in size and mode, so they are decoded
      and converted to RGB.

    - Raw mode decodes JPEGs from parquet shards when cache_dir is unset.
    - Latent mode reads precomputed VAE latents from safetensors shards.
    - Latent mode builds the cache on first use when the manifest is missing or
      the cache was built with a different VAE or image size.
    - Each image is cached once, flipped with flip_prob, to halve the cache size.
    - __getitem__ returns raw sample dicts in raw mode and (latent, label) in
      latent mode, where the latent is sampled and normalized by the VAE scale
      and bias.
    """

    def __init__(self, cfg_data) -> None:
        super().__init__(
            name=cfg_data.get("name", "dataset"),
            cache_dir=cfg_data.get("cache_dir", ""),
            image_transform=None,
            text_transform=None,
            sample_ratio=cfg_data.get("sample_ratio", 1.0),
            sample_seed=cfg_data.get("seed", 42),
        )
        self.path = cfg_data.path
        self.split = cfg_data.get("split", "train")
        self.img_size = cfg_data.get("img_size", 256)
        self.flip_prob = cfg_data.get("flip_prob", 0.)
        self.shared_cache = cfg_data.get("shared_cache", False)
        if self.cache_dir is not None:
            self.cache_dir = resolve_cache_dir(self.cache_dir, self.shared_cache)

        # Get data files.
        search_file = os.path.join(self.path, "data", f"{self.split}-*.parquet")
        self.files = sorted(glob.glob(search_file))

        # Get classes.
        self.class_text = load_class_text(self.path)

        # VAE latent normalization, applied to every latent returned by this dataset.
        self.vae_scale = cfg_data.get("vae_scale", 0.18215)
        self.vae_bias = cfg_data.get("vae_bias", 0.)

        # VAE-latent preprocessing knobs, only used when the cache is built on the fly.
        self.vae = cfg_data.get("vae")
        self.read_batch = cfg_data.get("read_batch", 256)
        self.vae_batch = cfg_data.get("vae_batch", 256)
        self.shard_size = cfg_data.get("shard_size", 8192)
        self.preprocess_workers = cfg_data.get("preprocess_workers", 8)

        if self.cache_dir is not None:
            self._init_latent_mode()
        else:
            self._init_raw_mode()

    # --------------------------------------------------------------------------------
    # Raw mode: return (image: Image, label text: str, label: int).
    # --------------------------------------------------------------------------------
    def _init_raw_mode(self) -> None:
        self.mode = "raw"

        # Global index -> (file, row_group) via row-group boundaries (metadata only).
        self._rg_starts: list[int] = []  # start index of row-group
        self._rg_loc: list[tuple[int, int]] = []  # (file index, in-file row-group index) of row group
        start = 0
        for fi, f in enumerate(self.files):
            meta = pq.ParquetFile(f).metadata
            for rg in range(meta.num_row_groups):
                self._rg_starts.append(start)
                self._rg_loc.append((fi, rg))
                start += meta.row_group(rg).num_rows
        self.num_sample = start

        # Per-worker lazy caches (kept empty here so they survive DataLoader fork).
        self._pf_cache: dict[int, pq.ParquetFile] = {}
        self._rg_key: tuple[int, int] | None = None
        self._rg_table: pa.Table | None = None
        
        self.build_indices(self.num_sample)

    def _row_group(self, file_idx: int, rg_idx: int) -> pa.Table:
        """Read a parquet row group, caching the most recent one per worker."""
        key = (file_idx, rg_idx)
        if self._rg_key != key:
            pf = self._pf_cache.get(file_idx)
            if pf is None:
                pf = pq.ParquetFile(self.files[file_idx])
                self._pf_cache[file_idx] = pf
            self._rg_table = pf.read_row_group(rg_idx, columns=["image", "label"])
            self._rg_key = key
        return self._rg_table

    def get_raw(self, index: int) -> dict[str, Any]:
        """Load raw data in raw mode."""
        if self.mode != "raw":
            raise RuntimeError("❌ get_raw is only available in raw mode (cache_dir unset)")
        rg = bisect.bisect_right(self._rg_starts, index) - 1  # find the row group index
        file_idx, rg_idx = self._rg_loc[rg]  # get the row group file and in-file row-group index
        local = index - self._rg_starts[rg]  # in-group index

        table = self._row_group(file_idx, rg_idx)
        image_bytes = table.column("image")[local].as_py()["bytes"]
        label = table.column("label")[local].as_py()
        raw_data = {
            "image": Image.open(io.BytesIO(image_bytes)).convert("RGB"),
            "text": self.class_text[label] if self.class_text else None,
            "label": label
        }
        return raw_data

    # --------------------------------------------------------------------------------
    # Latent mode: return (image VAE latent: float, label: int).
    # --------------------------------------------------------------------------------
    def _init_latent_mode(self) -> None:
        self.mode = "latent"
        if self._load_manifest() is None:
            self._build_cache()  # build cache if current cache is empty or incomplete
        manifest = self._load_manifest()
        if manifest is None:
            raise RuntimeError(f"❌ Latent cache build incomplete at {self.cache_dir}")

        self.shards: list[tuple[str, int]] = []
        start = 0
        for s in manifest["shards"]:
            path = os.path.join(self.cache_dir, s["file"])
            self.shards.append((path, start))
            start += s["num_samples"]
        self.num_sample = start
        self._shard_starts = [shard_start for _, shard_start in self.shards]
        self._handles: dict[str, Any] = {}  # per-worker lazy safe_open handles
        self.build_indices(start)

    def _load_manifest(self) -> dict[str, Any] | None:
        """Return the manifest if a complete cache exists for this split, else None."""
        manifest_path = os.path.join(self.cache_dir, f"manifest-{self.split}.json")
        if not os.path.exists(manifest_path):
            return None
        with open(manifest_path) as f:
            manifest = json.load(f)
        same_vae = os.path.basename(os.path.normpath(manifest["vae"])) == os.path.basename(os.path.normpath(self.vae))
        if not same_vae or manifest["img_size"] != self.img_size:
            print(f"⚠️ Latent cache at {self.cache_dir} was built with another VAE or image size; rebuilding")
            return None
        for s in manifest["shards"]:
            if not os.path.exists(os.path.join(self.cache_dir, s["file"])):
                return None
        return manifest

    def _verify_shared_cache(self, cache: str, run_id: str) -> bool:
        """Probe whether cache lives on a filesystem visible to every rank/node."""
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return False  # single process: a "shared" cache is identical to a local one
        probe = os.path.join(cache, f".probe-{run_id}")
        if dist.get_rank() == 0:
            open(probe, "w").close()
        dist.barrier()
        # A shared FS may lag slightly behind the writer; poll briefly before giving up.
        seen = False
        for _ in range(15):
            if os.path.exists(probe):
                seen = True
                break
            time.sleep(1.0)
        flag = torch.tensor([1 if seen else 0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if dist.get_rank() == 0:
            os.remove(probe)
        return bool(flag.item())

    def _build_cache(self) -> None:
        """
        Encode the parquet data to VAE latents and write a shard manifest.

        Two layouts, selected by self.shared_cache:
        - shared: all ranks across all nodes cooperate to build one cache on the
          shared path; global rank 0 writes the manifest.
        - local:  each node builds a complete cache on its own disk using that
          node's GPUs; each node's local rank 0 writes the manifest.
        If shared_cache is requested but the path is not visible to all nodes, it
        falls back to the local layout.
        """
        from diffusers.models import AutoencoderKL
        from safetensors.torch import save_file

        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "0")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        cache = self.cache_dir
        os.makedirs(cache, exist_ok=True)

        # Resolve the build-group layout: who cooperates on this cache directory.
        shared = self.shared_cache and self._verify_shared_cache(cache, run_id)
        if self.shared_cache and not shared:
            print("⚠️ shared_cache requested but the path is not visible to all nodes; "
                  "falling back to per-node local cache.")
        if shared:
            world = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world = torch.cuda.device_count()
            rank = dist.get_rank() % world

        manifest_path = os.path.join(cache, f"manifest-{self.split}.json")
        clean_flag = os.path.join(cache, f".clean-{self.split}-{run_id}")
        done_flag = os.path.join(cache, f".done-{self.split}-{run_id}-{rank:03d}")
        done_glob = os.path.join(cache, f".done-{self.split}-{run_id}-*")

        # Clean incomplete files.
        if rank == 0:
            print(f"💾 Building latent cache at {cache} ...")
            for pat in (f"latents-{self.split}-*.safetensors", f".done-{self.split}-*", f".clean-{self.split}-*"):
                for p in glob.glob(os.path.join(cache, pat)):
                    os.remove(p)
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
            open(clean_flag, "w").close()
        else:
            _wait_for(lambda: os.path.exists(clean_flag))

        # Load VAE.
        if self.vae is None:
            raise ValueError("❌ data.vae must be set to build the latent cache")
        vae = AutoencoderKL.from_pretrained(self.vae).to(device).eval()

        # Dataloader for parquet files.
        files = self.files[rank::world]
        stream = _ParquetImageStream(files, self.img_size, self.read_batch)
        loader = torch.utils.data.DataLoader(
            stream, batch_size=self.vae_batch, num_workers=self.preprocess_workers, pin_memory=True
        )

        # One progress bar per build group (driven by rank 0, tracking its own samples).
        num_local = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        bar = TqdmBar(num_local, "🚀 Encoding latents", "img", mininterval=30)

        out_m, out_l = [], []
        state = {"count": 0, "shard": 0}

        def write_shard() -> None:
            if not out_m:
                return
            tensors = {
                "moments": torch.cat(out_m).contiguous(),
                "label": torch.cat(out_l).contiguous(),
            }
            path = os.path.join(cache, f"latents-{self.split}-{rank:03d}-{state['shard']:05d}.safetensors")
            save_file(tensors, path, metadata={"img_size": str(self.img_size), "vae": self.vae})
            out_m.clear(); out_l.clear()
            state["count"] = 0
            state["shard"] += 1

        # Start encoding latents; flip each image with flip_prob and cache it once.
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            flip = torch.rand(imgs.shape[0], device=device) < self.flip_prob
            imgs[flip] = imgs[flip].flip(dims=[3])
            with torch.no_grad():
                out_m.append(vae.encode(imgs).latent_dist.parameters.cpu().half())
            out_l.append(labels.to(torch.int16))
            state["count"] += imgs.shape[0]
            bar.update(imgs.shape[0])
            if state["count"] >= self.shard_size:
                write_shard()
        write_shard()
        bar.close()
        del vae
        torch.cuda.empty_cache()

        # Mark this rank done; local rank 0 waits for all ranks, then writes the manifest.
        open(done_flag, "w").close()
        if rank == 0:
            _wait_for(lambda: len(glob.glob(done_glob)) >= world)
            _write_manifest(cache, self.split, self.img_size, self.vae)
            for p in glob.glob(done_glob):
                os.remove(p)
            os.remove(clean_flag)
        else:
            _wait_for(lambda: os.path.exists(manifest_path))

    def _get_latent(self, index: int) -> tuple[torch.Tensor, int]:
        index = self._indices[index]  # map filtered index back to the raw sample
        si = bisect.bisect_right(self._shard_starts, index) - 1
        path, start = self.shards[si]
        local = index - start

        f = self._handles.get(path)
        if f is None:
            f = safe_open(path, framework="pt")
            self._handles[path] = f

        label = int(f.get_slice("label")[local].item())

        # Sample the latent from its VAE posterior and normalize it here, so the
        # algorithm receives ready-to-use latents.
        mean, logvar = f.get_slice("moments")[local].float().chunk(2, dim=0)
        std = torch.exp(0.5 * logvar.clamp(-30., 20.))
        latent = mean + std * torch.randn_like(mean)
        return latent * self.vae_scale + self.vae_bias, label

    def __getitem__(self, index: int) -> dict[str, Any] | tuple[torch.Tensor, int]:
        if self.mode == "latent":
            return self._get_latent(index)
        return super().__getitem__(index)


# --------------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------------
def resolve_cache_dir(cache_dir: str, shared_cache: bool) -> str:
    """Return the effective cache dir for shared or node-local cache mode."""
    local_world = torch.cuda.device_count()
    if shared_cache or dist.get_world_size() <= local_world:
        print(f"📌 Use shared latent cache path: <{cache_dir}>")
        return cache_dir
    node = dist.get_rank() // local_world
    cache = os.path.join(cache_dir, f"node_{node:04d}")
    print(f"📌 Use node-local latent cache path: <{cache}>")
    return cache


def load_class_text(root: str) -> list[str] | None:
    """Load index->class-text list from the dataset's classes.py, or None."""
    path = os.path.join(root, "classes.py")
    if not os.path.exists(path):
        return None
    ns: dict[str, Any] = {}
    with open(path) as f:
        exec(f.read(), ns)
    return list(ns["IMAGENET2012_CLASSES"].values())


def _wait_for(condition: Callable[[], bool], timeout: float = 7200.0, interval: float = 2.0) -> None:
    """Poll until condition() is true, raising after timeout seconds."""
    start = time.monotonic()
    while not condition():
        if time.monotonic() - start > timeout:
            raise RuntimeError("❌ Timed out waiting for latent cache build coordination")
        time.sleep(interval)


def _center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop a PIL image to a square of image_size (DiT/MeanFlow style)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])


def _build_transform(img_size: int) -> Callable:
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Lambda(lambda img: _center_crop_arr(img, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _shard_num_samples(path: str) -> int:
    """Number of samples stored in a latent shard file."""
    with safe_open(path, framework="pt") as f:
        return f.get_slice("label").get_shape()[0]


def _write_manifest(cache_dir: str, split: str, img_size: int, vae_name: str) -> None:
    """Scan written shards and atomically write manifest-{split}.json."""
    shard_files = sorted(glob.glob(os.path.join(cache_dir, f"latents-{split}-*.safetensors")))
    shards: list[dict[str, Any]] = []
    total = 0
    for path in shard_files:
        n = _shard_num_samples(path)
        shards.append({"file": os.path.basename(path), "num_samples": n})
        total += n
    manifest = {
        "dataset": "imagenet-1k", "split": split, "img_size": img_size,
        "vae": vae_name, "num_classes": NUM_CLASSES, "num_samples": total,
        "latent_shape": [8, img_size // 8, img_size // 8], "shards": shards,
    }
    path = os.path.join(cache_dir, f"manifest-{split}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)
    print(f"📌 Wrote manifest: {total} samples across {len(shards)} shards -> <{path}>")


class _ParquetImageStream(torch.utils.data.IterableDataset):
    """
    Iterable parquet image stream for latent-cache building.

    - Splits files across DataLoader workers.
    - Decodes JPEGs and applies transforms in worker processes.
    - Yields (image_tensor, label) tuples for VAE encoding.
    """

    def __init__(self, 
        files: list[str],
        img_size: int,
        read_batch: int
    ) -> None:
        self.files = files
        self.img_size = img_size
        self.read_batch = read_batch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        files = self.files if info is None else self.files[info.id::info.num_workers]
        transform = _build_transform(self.img_size)
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=self.read_batch, columns=["image", "label"]):
                for img_struct, label in zip(batch.column("image").to_pylist(),
                                             batch.column("label").to_pylist()):
                    image = Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB")
                    yield transform(image), label
