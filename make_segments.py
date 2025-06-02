# make_segments.py
"""
Generate 1‑second / 50 %‑overlap segments for the SERL project using the
pre‑mixed Valentini‑Botinhao VoiceBank‑DEMAND corpus.

The script performs **only segmentation & dataset split re‑organisation** – no
additional mixing.  Output directory structure is created directly under the
Valentini root::

    valentini/
      train-0/{clean,noisy}/<spk>/*.wav
      val/{clean,noisy}/<spk>/*.wav          # 10 % of train‑0 (optional)
      test-0/{clean,noisy}/<spk>/*.wav
      train-1/{clean,noisy}/<spk>/*.wav
      test-1/{clean,noisy}/<spk>/*.wav

Each file is saved as 16‑bit PCM and named as
``p226_001_000.wav`` (original stem + segment index).

Default paths match the local Windows workspace but can be overridden via
command‑line arguments.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import torchaudio
torchaudio.set_audio_backend("soundfile")

# ----------------------------- constants ----------------------------------- #
ROOT_DEFAULT = (
    "C:/Users/AITER/Documents/Pythonworkspace/SE-RL/data/Valentini/valentini"
)

# sub‑directories shipped with the corpus
DIRS = {
    "clean56": "clean_trainset_56spk_wav",
    "clean28": "clean_trainset_28spk_wav",
    "clean_test": "clean_testset_wav",
    "noisy56": "noisy_trainset_56spk_wav",
    "noisy28": "noisy_trainset_28spk_wav",
    "noisy_test": "noisy_testset_wav",
}

SEG_LEN = 16_384  # 1 s @ 16 kHz
HOP_LEN = SEG_LEN // 2
SR = 16_000
PEAK_CLIP = 0.99

# ----------------------------- helpers ------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2))


def load_wav(path: Path) -> torch.Tensor:
    """
    Load mono waveform, resampling to 16 kHz if needed.
    Returns [1, T] float32.
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)

    if wav.dim() > 1:              # down-mix multi-channel
        wav = torch.mean(wav, dim=0)

    if sr != SR:                   # on-the-fly resample to 16 kHz
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, SR
        ).squeeze(0)
    return wav.unsqueeze(0)        # shape [1, T]



# --------------------------- split logic ----------------------------------- #

def speaker_ids(directory: Path) -> List[str]:
    """Return unique speaker prefixes inside a directory of wav files."""
    return sorted({p.stem.split("_")[0] for p in directory.glob("*.wav")})


def choose_test0_train0_extra(ids28: List[str], seed: int) -> Tuple[List[str], List[str]]:
    """Randomly choose 10 speakers for test‑0, remaining 18 for train‑0 extra."""
    rng = random.Random(seed)
    shuffled = ids28.copy()
    rng.shuffle(shuffled)
    return shuffled[:10], shuffled[10:]


# ------------------------- segmentation worker ----------------------------- #

def segment_worker(task: Tuple[str, Path, Path, Path]):
    """Process one original wav file -> segments to target dirs."""
    split_name, clean_path, noisy_path, out_root = task

    spk = clean_path.stem.split("_")[0]
    base_stem = clean_path.stem  # p226_001

    clean = load_wav(clean_path)
    noisy = load_wav(noisy_path)
    assert clean.shape == noisy.shape, "clean/noisy length mismatch"

    tgt_clean_dir = out_root / split_name / "clean" / spk
    tgt_noisy_dir = out_root / split_name / "noisy" / spk
    tgt_clean_dir.mkdir(parents=True, exist_ok=True)
    tgt_noisy_dir.mkdir(parents=True, exist_ok=True)

    total = clean.size(1)
    idx = 0
    for st in range(0, total - SEG_LEN + 1, HOP_LEN):
        en = st + SEG_LEN
        c_seg = clean[:, st:en]
        n_seg = noisy[:, st:en]
        fname = f"{base_stem}_{idx:03d}.wav"
        torchaudio.save(
            str(tgt_clean_dir / fname), c_seg, SR, encoding="PCM_S", bits_per_sample=16
        )
        torchaudio.save(
            str(tgt_noisy_dir / fname), n_seg, SR, encoding="PCM_S", bits_per_sample=16
        )
        idx += 1
    return idx  # number of segments


# --------------------------- main procedure -------------------------------- #

def build_tasks(root: Path, seed: int) -> Tuple[List[Tuple[str, Path, Path, Path]], Dict[str, List[str]]]:
    """Create segmentation tasks & return mapping split->segment paths for val selection."""
    paths: Dict[str, Path] = {k: root / v for k, v in DIRS.items()}

    # speaker id collections
    spk56 = speaker_ids(paths["clean56"])
    spk28 = speaker_ids(paths["clean28"])
    spk_test = speaker_ids(paths["clean_test"])

    test0_ids, train0_extra_ids = choose_test0_train0_extra(spk28, seed)

    train0_ids = spk56 + train0_extra_ids  # 56 + 18 = 74
    train1_ids = spk28                     # 28, overlaps allowed
    test1_ids = spk_test                   # 2

    split_map: Dict[str, List[Tuple[Path, Path]]] = {
        "train-0": [],
        "test-0": [],
        "train-1": [],
        "test-1": [],
    }

    # helper to append pairs
    def add_pairs(ids: List[str], clean_dir: Path, noisy_dir: Path, split_name: str):
        for spk in ids:
            for wav in clean_dir.glob(f"{spk}_*.wav"):
                split_map[split_name].append((wav, noisy_dir / wav.name))

    # populate pairs
    add_pairs(spk56, paths["clean56"], paths["noisy56"], "train-0")
    add_pairs(train0_extra_ids, paths["clean28"], paths["noisy28"], "train-0")
    add_pairs(test0_ids, paths["clean28"], paths["noisy28"], "test-0")
    add_pairs(train1_ids, paths["clean28"], paths["noisy28"], "train-1")
    add_pairs(test1_ids, paths["clean_test"], paths["noisy_test"], "test-1")

    tasks: List[Tuple[str, Path, Path, Path]] = []
    for split_name, pair_list in split_map.items():
        for clean_p, noisy_p in pair_list:
            tasks.append((split_name, clean_p, noisy_p, root))

    return tasks, split_map


def write_lst(root: Path, split_name: str, paths: List[Path]):
    lst_path = root / f"{split_name}.lst"
    rel_paths = [p.relative_to(root).as_posix() for p in sorted(paths)]
    with lst_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(rel_paths))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", default=ROOT_DEFAULT,
                        help="Valentini corpus root directory")
    parser.add_argument("--val_ratio", type=float, default=0.10,
                        help="Fraction of train‑0 segments moved to validation list")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    set_seed(args.seed)

    tasks, split_map = build_tasks(root, args.seed)

    # multiprocessing segmentation
    with mp.Pool(args.workers, initializer=set_seed, initargs=(args.seed,)) as pool:
        list(tqdm(pool.imap_unordered(segment_worker, tasks), total=len(tasks), desc="Segment"))

    # collect segment file paths per split for .lst generation
    seg_paths: Dict[str, List[Path]] = {k: [] for k in split_map.keys()}
    for split_name in split_map.keys():
        split_dir = root / split_name / "noisy"  # just inspect noisy side for path list
        seg_paths[split_name] = list(split_dir.rglob("*.wav"))

    # validation split (10 % of train‑0)
    all_train0 = seg_paths["train-0"]
    random.Random(args.seed).shuffle(all_train0)
    val_count = math.ceil(len(all_train0) * args.val_ratio)
    val_paths = all_train0[:val_count]
    seg_paths["val"] = val_paths
    seg_paths["train-0"] = all_train0[val_count:]

    # write lst files
    for split_name, paths in seg_paths.items():
        write_lst(root, split_name.replace("-", ""), paths)  # train-0 -> train0.lst / val -> val.lst

    print("\nDataset list summary:")
    for sn, plist in seg_paths.items():
        print(f"  {sn:7s}: {len(plist):6d} segments")

    print("\n✔ Segmentation & list generation complete.")


if __name__ == "__main__":
    main()
