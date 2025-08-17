#!/usr/bin/env python3
"""Create a tampered version of watermarked audio by adding noise at the end"""

import numpy as np
import soundfile as sf
import argparse

def create_tampered_audio(input_path, output_path, noise_duration_sec=10, noise_position="end"):
    """Create a tampered version by inserting noise"""
    
    print(f"📂 Reading original file: {input_path}")
    audio, sr = sf.read(input_path)
    
    # Ensure mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {len(audio)/sr:.2f} seconds")
    
    # Generate noise (pink noise - more natural sounding than white noise)
    noise_samples = int(noise_duration_sec * sr)
    
    # Generate pink noise (1/f noise) - sounds more like background noise
    white = np.random.randn(noise_samples)
    # Apply a simple low-pass filter to make it pink-ish
    pink = np.zeros_like(white)
    pink[0] = white[0]
    for i in range(1, len(white)):
        pink[i] = 0.95 * pink[i-1] + 0.05 * white[i]
    
    # Scale noise to be subtle but detectable (about 10% of typical audio amplitude)
    noise = pink * 0.1 * np.std(audio[:sr*5])  # Use first 5 seconds to estimate amplitude
    
    if noise_position == "end":
        # Add noise at the end
        tampered_audio = np.concatenate([audio, noise])
        print(f"✂️ Added {noise_duration_sec}s of noise at the end")
        
    elif noise_position == "middle":
        # Insert noise in the middle
        mid_point = len(audio) // 2
        tampered_audio = np.concatenate([
            audio[:mid_point],
            noise,
            audio[mid_point:]
        ])
        print(f"✂️ Inserted {noise_duration_sec}s of noise in the middle")
        
    elif noise_position == "replace_end":
        # Replace last 10 seconds with noise
        if len(audio) > noise_samples:
            tampered_audio = np.concatenate([
                audio[:-noise_samples],
                noise
            ])
            print(f"✂️ Replaced last {noise_duration_sec}s with noise")
        else:
            print("⚠️ Audio too short to replace end, adding noise instead")
            tampered_audio = np.concatenate([audio, noise])
    
    print(f"   New duration: {len(tampered_audio)/sr:.2f} seconds")
    
    # Save tampered version
    sf.write(output_path, tampered_audio, sr, subtype='PCM_16')
    print(f"✅ Saved tampered version to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create tampered audio for testing")
    parser.add_argument("--input", default="phone_recording_id_12345.wav", 
                        help="Input watermarked file")
    parser.add_argument("--output", default="phone_recording_id_12345_tampered.wav",
                        help="Output tampered file")
    parser.add_argument("--noise-duration", type=float, default=10,
                        help="Duration of noise in seconds")
    parser.add_argument("--position", choices=["end", "middle", "replace_end"], 
                        default="end",
                        help="Where to add the noise")
    
    args = parser.parse_args()
    
    create_tampered_audio(
        args.input,
        args.output,
        args.noise_duration,
        args.position
    )

if __name__ == "__main__":
    main()