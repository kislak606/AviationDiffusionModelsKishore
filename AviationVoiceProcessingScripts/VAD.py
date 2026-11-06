#!/usr/bin/env python3
"""
Cut long ATC radio recordings into timestamped speech clips using Silero VAD.

Layout expected (example):
    SRC/
      adsb/                 <- skipped
      knuq_gnd_twr/*.mp3
      koak_del_gnd_twr/*.mp3
      ...
      metar/                <- skipped

Output mirrors the folder structure:
    OUT/knuq_gnd_twr/<recording_stem>/<recording_stem>__0003_00123450.flac
    OUT/knuq_gnd_twr/<recording_stem>/segments.csv     (per-recording manifest, doubles as "done" marker)
    OUT/manifest.csv                                   (all recordings merged)

The number at the end of each clip name is its start offset in the source
recording, in milliseconds, so clips stay time-aligned to the original.

Requirements:
    pip install silero-vad soundfile numpy
    ffmpeg on PATH   (Windows: `winget install ffmpeg`)

Usage:
    # 1) tune on a small sample first (3 recordings per folder, nothing written):
    python vad_clip.py "D:\\atc" "D:\\atc_clips" --limit-per-folder 3 --dry-run

    # 2) full run (resumable - re-run the same command after an interruption):
    python vad_clip.py "D:\\atc" "D:\\atc_clips"
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000  # Silero VAD native rate; clips are saved at this rate (mono)
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
DEFAULT_EXCLUDE = "adsb,metar"
# Real ATC speech clips measured RMS >= ~0.024; dead recordings sit at ~0.0001
DEFAULT_MIN_RMS = 0.005
DEFAULT_MIN_PEAK = 0.01
FIELDS = ["clip_file", "source_file", "facility", "start_s", "end_s",
          "duration_s", "clip_start_utc"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def decode(path):
    """Decode any audio file to 16 kHz mono float32 via ffmpeg
    (falls back to soundfile if ffmpeg isn't installed)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return resample(data.mean(axis=1), sr, SR)
    return np.frombuffer(out, dtype=np.float32).copy()


def resample(x, sr_in, sr_out):
    """Chunked FFT resample (no scipy/torchaudio needed)."""
    if sr_in == sr_out:
        return x
    import torch
    g = math.gcd(sr_in, sr_out)
    chunk_in, chunk_out = sr_in // g * 2000, sr_out // g * 2000
    xt = torch.nn.functional.pad(torch.from_numpy(x), (0, (-len(x)) % chunk_in))
    outs = []
    for i in range(0, len(xt), chunk_in):
        spec = torch.fft.rfft(xt[i:i + chunk_in])
        n_bins = chunk_out // 2 + 1
        spec = spec[:n_bins] if n_bins <= len(spec) else torch.nn.functional.pad(
            spec, (0, n_bins - len(spec)))
        outs.append(torch.fft.irfft(spec, n=chunk_out) * (chunk_out / chunk_in))
    y = torch.cat(outs).numpy()
    return y[: int(round(len(x) * sr_out / sr_in))].astype(np.float32)


def parse_start_time(stem):
    """Best-effort recording start time from a file name (assumed UTC).
    Handles LiveATC style  KSFO-Twr-Nov-06-2026-2000Z  and
    20261106_200000 / 2026-11-06-2000 style names. Returns datetime or None.
    Adjust here if your files are named differently."""
    m = re.search(r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})(\d{2})Z", stem)
    if m:
        try:
            return datetime.strptime(
                f"{m[1]} {m[2]} {m[3]} {m[4]}{m[5]}", "%b %d %Y %H%M"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})[-_T ]?(\d{2})[-_:]?(\d{2})(?:[-_:]?(\d{2}))?", stem)
    if m:
        try:
            y, mo, d, h, mi = (int(m[i]) for i in range(1, 6))
            s = int(m[6]) if m[6] else 0
            return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)  # parallelism comes from worker processes
        from silero_vad import load_silero_vad
        _model = load_silero_vad()
    return _model


# --------------------------------------------------------------------------
# per-recording worker
# --------------------------------------------------------------------------
def process_file(job):
    src, facility, out_dir, cfg = job
    import torch
    from silero_vad import get_speech_timestamps

    torch.set_num_threads(1)  # before decode(): its resampler also uses torch
    src = Path(src)
    stem = src.stem
    audio = decode(src)
    total_s = len(audio) / SR

    # Dead recordings (decoder noise floor only) skip the VAD entirely
    if len(audio) == 0 or float(np.abs(audio).max()) < cfg["min_peak"]:
        spans = []
    else:
        spans = get_speech_timestamps(
            torch.from_numpy(audio), get_model(), sampling_rate=SR,
            threshold=cfg["threshold"],
            min_speech_duration_ms=cfg["min_speech_ms"],
            min_silence_duration_ms=cfg["min_silence_ms"],
            speech_pad_ms=cfg["pad_ms"],
            max_speech_duration_s=cfg["max_clip_s"],
        )
        # Drop clips with no real audio (VAD false positives on near-silence)
        spans = [sp for sp in spans
                 if float(np.sqrt(np.mean(audio[sp["start"]:sp["end"]] ** 2)))
                 >= cfg["min_rms"]]

    t0 = parse_start_time(stem)
    clip_dir = Path(out_dir) / facility / stem
    rows = []
    for i, sp in enumerate(spans):
        s, e = sp["start"], sp["end"]
        start_s, end_s = s / SR, e / SR
        name = f"{stem}__{i:04d}_{int(round(start_s * 1000)):08d}.flac"
        rel = f"{facility}/{stem}/{name}"
        if not cfg["dry_run"]:
            clip_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(Path(out_dir) / rel), audio[s:e], SR,
                     format="FLAC", subtype="PCM_16")
        rows.append({
            "clip_file": rel,
            "source_file": f"{facility}/{src.name}",
            "facility": facility,
            "start_s": f"{start_s:.3f}",
            "end_s": f"{end_s:.3f}",
            "duration_s": f"{end_s - start_s:.3f}",
            "clip_start_utc": (t0 + timedelta(seconds=start_s)).isoformat()
            if t0 else "",
        })

    if cfg.get("preview") and spans and not cfg["dry_run"]:
        gap = np.zeros(int(0.3 * SR), dtype=np.float32)
        parts = [p for sp in spans for p in (audio[sp["start"]:sp["end"]], gap)]
        sf.write(str(clip_dir / f"{stem}__speech_only.flac"),
                 np.concatenate(parts), SR, format="FLAC", subtype="PCM_16")

    if not cfg["dry_run"]:
        # per-recording manifest written LAST, atomically => done marker
        clip_dir.mkdir(parents=True, exist_ok=True)
        tmp = clip_dir / "segments.csv.tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, clip_dir / "segments.csv")

    speech_s = sum(float(r["duration_s"]) for r in rows)
    return str(src), total_s, speech_s, len(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def discover(src_root, exclude, limit_per_folder):
    src_root = Path(src_root)
    if src_root.is_file():  # single-recording mode
        return [(src_root, src_root.parent.name)]
    jobs = []
    for d in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if d.name.lower() in exclude:
            continue
        files = sorted(f for f in d.rglob("*")
                       if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        if limit_per_folder:
            files = files[:limit_per_folder]
        jobs += [(f, d.name) for f in files]
    return jobs


def merge_manifests(out_dir):
    out_dir = Path(out_dir)
    n = 0
    with open(out_dir / "manifest.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        for seg in sorted(out_dir.glob("*/*/segments.csv")):
            with open(seg, newline="") as fi:
                for row in csv.DictReader(fi):
                    w.writerow(row)
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="root folder containing one sub-folder per feed, "
                                "or a single audio file")
    ap.add_argument("out", help="output folder for clips + manifest")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                    help="comma-separated folder names to skip (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="speech probability threshold; lower = more sensitive")
    ap.add_argument("--min-speech-ms", type=int, default=250,
                    help="drop detections shorter than this (squelch clicks)")
    ap.add_argument("--min-silence-ms", type=int, default=800,
                    help="silence longer than this splits clips; shorter gaps are merged")
    ap.add_argument("--pad-ms", type=int, default=300,
                    help="context kept before/after each clip")
    ap.add_argument("--max-clip-s", type=float, default=60.0)
    ap.add_argument("--min-rms", type=float, default=DEFAULT_MIN_RMS,
                    help="discard clips whose RMS (full scale = 1.0) is below this")
    ap.add_argument("--min-peak", type=float, default=DEFAULT_MIN_PEAK,
                    help="skip recordings whose peak amplitude is below this")
    ap.add_argument("--limit-per-folder", type=int, default=0,
                    help="only process the first N recordings per folder (for tuning)")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect only; print stats, write nothing")
    a = ap.parse_args()

    exclude = {x.strip().lower() for x in a.exclude.split(",") if x.strip()}
    cfg = dict(threshold=a.threshold, min_speech_ms=a.min_speech_ms,
               min_silence_ms=a.min_silence_ms, pad_ms=a.pad_ms,
               max_clip_s=a.max_clip_s, min_rms=a.min_rms, min_peak=a.min_peak,
               dry_run=a.dry_run,
               preview=Path(a.src).is_file())

    found = discover(a.src, exclude, a.limit_per_folder)
    todo = []
    for f, fac in found:
        done = Path(a.out) / fac / f.stem / "segments.csv"
        if not a.dry_run and done.exists():
            continue
        todo.append((str(f), fac, a.out, cfg))
    print(f"{len(found)} recordings found, {len(todo)} to process "
          f"({len(found) - len(todo)} already done), {a.workers} workers",
          flush=True)

    per_fac = defaultdict(lambda: [0.0, 0.0, 0])
    failures = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_file, j): j for j in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                src, total, speech, n = fut.result()
            except Exception as exc:  # keep going; report at end
                failures.append((j[0], repr(exc)))
                print(f"[{k}/{len(todo)}] FAILED {j[0]}: {exc!r}", flush=True)
                continue
            p = per_fac[j[1]]
            p[0] += total; p[1] += speech; p[2] += n
            print(f"[{k}/{len(todo)}] {Path(src).name}: {n} clips, "
                  f"{speech / 60:.1f} of {total / 60:.1f} min kept "
                  f"({100 * speech / max(total, 1e-9):.1f}%)", flush=True)

    print("\nPer-folder summary (this run):")
    for fac, (tot, sp, n) in sorted(per_fac.items()):
        print(f"  {fac:32s} {n:6d} clips  {sp / 3600:6.2f} h speech of "
              f"{tot / 3600:7.2f} h  ({100 * sp / max(tot, 1e-9):.1f}%)")

    if not a.dry_run:
        n = merge_manifests(a.out)
        print(f"\nWrote {Path(a.out) / 'manifest.csv'} ({n} clips total)")
    if failures:
        print(f"\n{len(failures)} recordings failed (re-run to retry):")
        for f, e in failures:
            print(f"  {f}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()