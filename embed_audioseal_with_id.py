import soundfile as sf
import torch
import numpy as np
from audioseal import AudioSeal
import argparse

SR_EXPECTED = 16000
CHUNK_SEC = 30
PRINT_EVERY = 1

def to_tensor_mono(x: np.ndarray):
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    t = torch.from_numpy(x).to(torch.float32)
    return t.unsqueeze(0).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="hearing.raw.wav", help="Input WAV file")
    ap.add_argument("--out", dest="out_path", default="hearing.wm_with_id.wav", help="Output watermarked WAV")
    ap.add_argument("--manifest-id", type=int, default=12345, help="16-bit manifest ID to embed (0-65535)")
    args = ap.parse_args()
    
    if not 0 <= args.manifest_id <= 65535:
        print(f"Error: manifest-id must be between 0 and 65535, got {args.manifest_id}")
        return
    
    print(f"Embedding watermark with manifest ID: {args.manifest_id}")
    
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    
    # Create message tensor for the manifest ID
    # AudioSeal expects a 16-bit message as a tensor
    message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
    # Convert manifest_id to binary and set the bits
    for i in range(16):
        bit = (args.manifest_id >> i) & 1
        message[0, i] = bit
    
    with sf.SoundFile(args.in_path, "r") as fin:
        assert fin.samplerate == SR_EXPECTED, f"Expected {SR_EXPECTED} Hz, got {fin.samplerate}"
        n_samples = len(fin)
        chunk_size = CHUNK_SEC * SR_EXPECTED
        
        with sf.SoundFile(args.out_path, "w", samplerate=SR_EXPECTED, channels=1, subtype="PCM_16") as fout:
            chunk_idx = 0
            written = 0
            while True:
                audio = fin.read(frames=chunk_size, dtype="float32", always_2d=False)
                if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
                    break
                t = to_tensor_mono(audio)
                with torch.no_grad():
                    # Pass the message to embed the manifest ID
                    wm = gen.get_watermark(t, SR_EXPECTED, message=message)
                    watermarked = (t + wm).squeeze().cpu().numpy()
                fout.write(watermarked)
                written += watermarked.shape[0]
                chunk_idx += 1
                if chunk_idx % PRINT_EVERY == 0:
                    secs = written / SR_EXPECTED
                    print(f"Chunk {chunk_idx} done — ~{secs/60:.1f} min processed")
    
    print(f"✅ Done: {args.out_path}")
    print(f"✅ Embedded manifest ID: {args.manifest_id}")

if __name__ == "__main__":
    main()