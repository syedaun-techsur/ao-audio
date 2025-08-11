import soundfile as sf
import torch
import numpy as np
from audioseal import AudioSeal

IN_PATH = "hearing.raw.wav"
OUT_PATH = "hearing.wm.wav"
SR_EXPECTED = 16000
CHUNK_SEC = 30         # try 30; increase to 60 if stable
PRINT_EVERY = 1        # progress print frequency (in chunks)

def to_tensor_mono(x: np.ndarray):
    # x shape: (samples,) or (samples, channels)
    if x.ndim == 2:
        x = np.mean(x, axis=1)  # force mono if needed
    t = torch.from_numpy(x).to(torch.float32)
    return t.unsqueeze(0).unsqueeze(0)  # (1,1,T)

def main():
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))  # be gentle on CPU
    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    with sf.SoundFile(IN_PATH, "r") as fin:
        assert fin.samplerate == SR_EXPECTED, f"Expected {SR_EXPECTED} Hz, got {fin.samplerate}"
        n_samples = len(fin)
        chunk_size = CHUNK_SEC * SR_EXPECTED
        with sf.SoundFile(OUT_PATH, "w", samplerate=SR_EXPECTED, channels=1, subtype="PCM_16") as fout:
            chunk_idx = 0
            written = 0
            while True:
                audio = fin.read(frames=chunk_size, dtype="float32", always_2d=False)
                if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
                    break
                t = to_tensor_mono(audio)
                with torch.no_grad():
                    wm = gen.get_watermark(t, SR_EXPECTED)
                    watermarked = (t + wm).squeeze().cpu().numpy()
                fout.write(watermarked)
                written += watermarked.shape[0]
                chunk_idx += 1
                if chunk_idx % PRINT_EVERY == 0:
                    secs = written / SR_EXPECTED
                    print(f"Chunk {chunk_idx} done — ~{secs/60:.1f} min processed")
    print("✅ Done:", OUT_PATH)

if __name__ == "__main__":
    main()

