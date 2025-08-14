import argparse
import numpy as np
import soundfile as sf
import torch
from audioseal import AudioSeal

SR_EXPECTED = 16000
CHUNK_SEC = 10          # analyze in 10s windows 
FRAME_THRESH = 0.5      # per-frame watermark presence threshold 
MIN_GAP_SEC = 2.0       # report gaps >= 2s as suspicious 

def to_tensor_mono(x: np.ndarray):
    if x.ndim == 2:
        x = np.mean(x, axis=1)  # downmix stereo -> mono
    t = torch.from_numpy(x).to(torch.float32)
    return t.unsqueeze(0).unsqueeze(0)      # (1,1,T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="hearing.wm.wav", help="Input WAV (16 kHz mono)")
    ap.add_argument("--chunk-sec", type=float, default=CHUNK_SEC, help="Chunk size in seconds")
    ap.add_argument("--min-gap-sec", type=float, default=MIN_GAP_SEC, help="Min unwatermarked span to report")
    args = ap.parse_args()

    # Load detector once
    detector = AudioSeal.load_detector("audioseal_detector_16bits")

    with sf.SoundFile(args.in_path, "r") as fin:
        assert fin.samplerate == SR_EXPECTED, f"Expected {SR_EXPECTED} Hz, got {fin.samplerate}"
        total_samples = len(fin)
        chunk_size = int(args.chunk_sec * SR_EXPECTED)

        total_frames = 0
        total_pos_frames = 0
        gaps = []  # list of (start_sec, end_sec) where watermark frames are mostly absent

        cursor_samples = 0
        chunk_idx = 0

        while True:
            audio = fin.read(frames=chunk_size, dtype="float32", always_2d=False)
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                break

            t = to_tensor_mono(audio)

            # Low-level call gives framewise probabilities; [:,1,:] = "watermarked" prob (per README)
            with torch.no_grad():
                frame_probs, msg_probs = detector(t, SR_EXPECTED)  # frame_probs: (B,2,frames)
            wm_probs = frame_probs[:, 1, :].squeeze(0).cpu().numpy()  # (frames,)

            # Aggregate for overall score
            total_frames += wm_probs.shape[0]
            total_pos_frames += np.sum(wm_probs >= FRAME_THRESH)

            # Find long regions under threshold inside this chunk
            below = wm_probs < FRAME_THRESH
            if np.any(below):
                # Convert contiguous runs of "below" into time spans
                idx = np.flatnonzero(np.diff(np.concatenate(([0], below.view(np.int8), [0]))))
                runs = list(zip(idx[0::2], idx[1::2]))  # (start_idx, end_idx)
                for s, e in runs:
                    dur_sec = (e - s) / SR_EXPECTED
                    if dur_sec >= args.min_gap_sec:
                        start_sec = (cursor_samples + s) / SR_EXPECTED
                        end_sec = (cursor_samples + e) / SR_EXPECTED
                        gaps.append((start_sec, end_sec))

            cursor_samples += audio.shape[0]
            chunk_idx += 1
            # Suppress verbose progress output - only show final results

        overall_pct = 100.0 * (total_pos_frames / max(1, total_frames))
        duration_min = total_samples / SR_EXPECTED / 60.0
        print("\n===== AudioSeal Detection Summary =====")
        print(f"File: {args.in_path}")
        print(f"Duration: {duration_min:.2f} min at {SR_EXPECTED} Hz")
        print(f"Overall watermark-positive frames: {overall_pct:.2f}%")

        if gaps:
            print("\n⚠️  Regions with low/no watermark (>= {:.1f}s):".format(args.min_gap_sec))
            for (s, e) in gaps:
                print(f" - {s:8.2f}s → {e:8.2f}s  (len {(e-s):.2f}s)")
        else:
            print("\nNo long unwatermarked gaps found.")
        
        # Optional: high-level API also available (single probability + decoded 16-bit message)
        # result, message = detector.detect_watermark( ... , SR_EXPECTED)
        # print("High-level probability:", float(result))
        # print("Decoded message bits:", message)

if __name__ == "__main__":
    main()
