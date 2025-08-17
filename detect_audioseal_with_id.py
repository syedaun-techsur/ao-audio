import argparse
import numpy as np
import soundfile as sf
import torch
from audioseal import AudioSeal

SR_EXPECTED = 16000

def to_tensor_mono(x: np.ndarray):
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    t = torch.from_numpy(x).to(torch.float32)
    return t.unsqueeze(0).unsqueeze(0)

def decode_message_to_id(message_tensor):
    """Convert the decoded message tensor back to manifest ID"""
    if message_tensor is None:
        return None
    
    # message_tensor should be shape (1, 16) with binary values
    message_np = message_tensor.cpu().numpy().flatten()
    manifest_id = 0
    for i in range(16):
        if i < len(message_np):
            bit = int(message_np[i] > 0.5)  # Threshold to get binary value
            manifest_id |= (bit << i)
    return manifest_id

def get_mock_manifest_from_id(manifest_id):
    """Mock function to simulate retrieving C2PA manifest from watermark ID"""
    manifests_db = {
        12345: {
            "creator": "Official Court Recording",
            "date": "2024-01-15",
            "court_id": "USDC-2024-001",
            "title": "Hearing on Case No. 2024-CV-00123",
            "recorder": "Court Reporter Jane Smith",
            "location": "US District Court, Southern District",
            "hash": "sha256:a3f5b8c9d2e1f7a4b6c8e3d9f1a2b7c5"
        },
        54321: {
            "creator": "Official Court Recording",
            "date": "2024-02-20",
            "court_id": "USDC-2024-047",
            "title": "Preliminary Hearing - State v. Johnson",
            "recorder": "Court Reporter John Doe",
            "location": "US District Court, Northern District",
            "hash": "sha256:b7d3a1c5e9f4d2a8c6b1e7f3a9d5c2b8"
        },
        1234: {
            "creator": "Test Recording",
            "date": "2024-01-10",
            "court_id": "TEST-2024-001",
            "title": "Test Audio File",
            "recorder": "Test System",
            "location": "Test Environment",
            "hash": "sha256:test123456789abcdef"
        }
    }
    return manifests_db.get(manifest_id, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input WAV (16 kHz mono)")
    args = ap.parse_args()
    
    print(f"\n🔍 Detecting watermark and extracting manifest ID from: {args.in_path}")
    
    # Load detector
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    
    with sf.SoundFile(args.in_path, "r") as fin:
        if fin.samplerate != SR_EXPECTED:
            print(f"Error: Expected {SR_EXPECTED} Hz, got {fin.samplerate} Hz")
            return
        
        # Read the entire file (or first 30 seconds for testing)
        max_duration_sec = 30
        max_samples = min(len(fin), max_duration_sec * SR_EXPECTED)
        audio = fin.read(frames=max_samples, dtype="float32", always_2d=False)
        
        if audio.size == 0:
            print("Error: Empty audio file")
            return
        
        t = to_tensor_mono(audio)
        
        # Use high-level API to get both detection result and message
        with torch.no_grad():
            result, message = detector.detect_watermark(t, SR_EXPECTED)
        
        detection_prob = float(result)
        print(f"\n📊 Watermark Detection Results:")
        print(f"   Detection Probability: {detection_prob:.2%}")
        
        if detection_prob > 0.5:
            print(f"   ✅ Watermark DETECTED")
            
            # Decode the message to get manifest ID
            if message is not None:
                manifest_id = decode_message_to_id(message)
                print(f"   📋 Extracted Manifest ID: {manifest_id}")
                
                # Look up the manifest
                manifest = get_mock_manifest_from_id(manifest_id)
                if manifest:
                    print(f"\n🔗 Retrieved C2PA Manifest from ID {manifest_id}:")
                    print(f"   Creator: {manifest['creator']}")
                    print(f"   Date: {manifest['date']}")
                    print(f"   Court ID: {manifest['court_id']}")
                    print(f"   Title: {manifest['title']}")
                    print(f"   Recorder: {manifest['recorder']}")
                    print(f"   Location: {manifest['location']}")
                    print(f"   Hash: {manifest['hash']}")
                else:
                    print(f"   ⚠️ No manifest found for ID {manifest_id}")
            else:
                print(f"   ⚠️ No message decoded from watermark")
        else:
            print(f"   ❌ No watermark detected (probability below threshold)")

if __name__ == "__main__":
    main()