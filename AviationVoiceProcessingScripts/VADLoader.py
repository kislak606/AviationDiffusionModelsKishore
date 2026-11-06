#!/usr/bin/env python3
"""
Run VAD clipping over the whole raw ATC dataset (~130 GB of mp3s).

    python VADLoader.py                  # full run -> D:\\VADProcessedOutputAudio
    python VADLoader.py --workers 16     # override worker count
    python VADLoader.py --limit-per-folder 3 --out D:\\vad_test   # quick trial

Resumable: recordings that already have a segments.csv in the output are
skipped, so just re-run the same command after a crash/reboot. Progress, ETA and
failures are appended to <out>/vad_run.log, so it can run unattended.
All clipping logic lives in VAD.py.
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import VAD

DEFAULT_SRC = r"D:\ScrapedData\20260401-raw_dataset_v1"
DEFAULT_OUT = r"D:\VADProcessedOutputAudio"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--exclude", default=VAD.DEFAULT_EXCLUDE)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="default: logical cores minus 2")
    ap.add_argument("--limit-per-folder", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech-ms", type=int, default=250)
    ap.add_argument("--min-silence-ms", type=int, default=800)
    ap.add_argument("--pad-ms", type=int, default=300)
    ap.add_argument("--max-clip-s", type=float, default=60.0)
    ap.add_argument("--min-rms", type=float, default=VAD.DEFAULT_MIN_RMS)
    ap.add_argument("--min-peak", type=float, default=VAD.DEFAULT_MIN_PEAK)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "vad_run.log", "a", buffering=1, encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    cfg = dict(threshold=a.threshold, min_speech_ms=a.min_speech_ms,
               min_silence_ms=a.min_silence_ms, pad_ms=a.pad_ms,
               max_clip_s=a.max_clip_s, min_rms=a.min_rms, min_peak=a.min_peak,
               dry_run=False, preview=False)

    exclude = {x.strip().lower() for x in a.exclude.split(",") if x.strip()}
    found = VAD.discover(a.src, exclude, a.limit_per_folder)
    todo = [(str(f), fac, str(out), cfg) for f, fac in found
            if not (out / fac / f.stem / "segments.csv").exists()]
    log(f"START {len(found)} recordings found, {len(todo)} to process "
        f"({len(found) - len(todo)} already done), {a.workers} workers -> {out}")

    per_fac = defaultdict(lambda: [0.0, 0.0, 0])
    empty = failed = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(VAD.process_file, j): j for j in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                src, total, speech, n = fut.result()
            except Exception as exc:
                failed += 1
                log(f"FAILED {j[0]}: {exc!r}")
                continue
            p = per_fac[j[1]]
            p[0] += total; p[1] += speech; p[2] += n
            empty += n == 0
            if k % 50 == 0 or k == len(todo):
                el = time.time() - t_start
                eta = el / k * (len(todo) - k)
                log(f"[{k}/{len(todo)}] {100 * k / len(todo):.1f}% | "
                    f"{empty} empty, {failed} failed | "
                    f"elapsed {el / 3600:.2f} h, ETA {eta / 3600:.2f} h")

    log("Per-folder summary (this run):")
    for fac, (tot, sp, n) in sorted(per_fac.items()):
        log(f"  {fac:32s} {n:7d} clips  {sp / 3600:7.2f} h speech of "
            f"{tot / 3600:8.2f} h ({100 * sp / max(tot, 1e-9):.1f}%)")
    n = VAD.merge_manifests(out)
    log(f"DONE manifest.csv has {n} clips total; {empty} empty recordings, "
        f"{failed} failed (re-run to retry failures)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
