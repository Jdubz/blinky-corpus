#!/usr/bin/env python3
"""Features for scripts/drum_suspect.py, run on the Spark as a gpu_task (CPU librosa is enough).

Inputs (env): SPARK_INPUT_STEMS (htdemucs drum stems, <jobid>/drums.wav), SPARK_INPUT_MIXA/B (tars
of the edm mp3s); tracks.json beside this file maps each track to its stem and mix.
Per track, into SPARK_OUTPUT_FEATS/<track>.npz:
  centroid_mix, rolloff_mix, centroid_drums, rolloff_drums  spectral centroid / 85% rolloff (Hz), 10 ms hop
  rms_mix, rms_drums                                         RMS, 10 ms hop
  os_{low,mid,high}_{mix,drums}  reviewer-style band onset strength (zero-phase Butterworth band,
                                 1 ms RMS blocks, log1p(100x) of the 99.9th-pct-normalised envelope,
                                 positive centred slope), 1 ms hop, float16
  tempo_librosa, beats_librosa   librosa.feature.tempo / beat_track (Ellis 2007) on the drum stem
"""
import json, os, sys, tarfile, signal
from pathlib import Path
import numpy as np

SR = 44100
BANDS = {"low": (30, 150), "mid": (150, 5000), "high": (5000, 16000)}
HERE = Path(__file__).resolve().parent


def band_os(y, lo, hi, h):
    from scipy.signal import butter, sosfiltfilt
    if hi >= SR / 2 * 0.95:
        sos = butter(4, lo, btype="highpass", fs=SR, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    x = sosfiltfilt(sos, y.astype(np.float64))
    hop = SR // 1000
    nb = len(x) // hop
    e = (x[: nb * hop].reshape(nb, hop) ** 2).sum(1)
    w = 2 * h + 1
    env = np.sqrt(np.maximum(np.convolve(e, np.ones(w), mode="same"), 0) / (w * hop))
    ref = np.percentile(env, 99.9) or 1.0
    c = np.log1p(100 * np.minimum(env / ref, 1.5)) / np.log1p(100)
    d = 3 if lo < 150 else 1
    os_ = np.zeros(nb)
    os_[d:-d] = np.maximum(0, c[2 * d:] - c[: -2 * d])
    return os_.astype(np.float16)


def main():
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(42))
    import librosa, soundfile as sf
    tracks = json.loads((HERE / "tracks.json").read_text())
    out = Path(os.environ["SPARK_OUTPUT_FEATS"])
    work = Path(os.environ.get("SPARK_CHECKPOINT_DIR", "/work/ck")) / "mix"
    work.mkdir(parents=True, exist_ok=True)
    for t in ("mixA", "mixB"):
        src = next(Path(os.environ[f"SPARK_INPUT_{t.upper()}"]).rglob(f"{t}.tar"))
        with tarfile.open(src) as tf:
            tf.extractall(work, filter="data")
    stems = Path(os.environ["SPARK_INPUT_STEMS"])
    n = len(tracks)
    for i, (name, t) in enumerate(sorted(tracks.items()), 1):
        mix, _ = librosa.load(str(work / t["mix"]), sr=SR, mono=True)
        d, sr = sf.read(str(stems / t["stem"]), always_2d=True, dtype="float32")
        drums = d.mean(1)
        feats = {}
        for tag, y in (("mix", mix), ("drums", drums)):
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=441))
            feats[f"centroid_{tag}"] = librosa.feature.spectral_centroid(S=S, sr=SR)[0].astype(np.float32)
            feats[f"rolloff_{tag}"] = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0].astype(np.float32)
            feats[f"rms_{tag}"] = librosa.feature.rms(S=S, frame_length=2048)[0].astype(np.float32)
            for b, (lo, hi) in BANDS.items():
                feats[f"os_{b}_{tag}"] = band_os(y, lo, hi, 10 if b == "low" else 1)
        oenv = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=441)
        tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=441, units="time")
        feats["tempo_librosa"] = np.atleast_1d(np.asarray(tempo, dtype=np.float32))
        feats["beats_librosa"] = np.asarray(beats, dtype=np.float32)
        np.savez_compressed(out / f"{name}.npz", **feats)
        print(f"STEP {i}/{n} {name} tempo {float(np.atleast_1d(tempo)[0]):.2f} beats {len(beats)}", flush=True)
    (out / "meta.json").write_text(json.dumps({"librosa": librosa.__version__}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
