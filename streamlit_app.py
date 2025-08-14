import streamlit as st
import subprocess
import json
import numpy as np
import soundfile as sf
import torch
from audioseal import AudioSeal
import os
from pathlib import Path

SR_EXPECTED = 16000
CHUNK_SEC = 10
FRAME_THRESH = 0.5
MIN_GAP_SEC = 1.0

st.set_page_config(
    page_title="Court Audio Verification System",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Court Audio Verification System")
st.markdown("### Verify authenticity of court hearing recordings using C2PA and AudioSeal")

def run_c2patool(file_path):
    """Run c2patool --info on a file and return the output"""
    try:
        result = subprocess.run(
            ["c2patool", file_path, "--info"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error running c2patool: {str(e)}"

def to_tensor_mono(x: np.ndarray):
    """Convert audio to mono tensor for AudioSeal"""
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    t = torch.from_numpy(x).to(torch.float32)
    return t.unsqueeze(0).unsqueeze(0)

def check_watermark(file_path):
    """Check AudioSeal watermark on a file"""
    try:
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        
        with sf.SoundFile(file_path, "r") as fin:
            if fin.samplerate != SR_EXPECTED:
                return {
                    "status": "ERROR",
                    "error": f"Expected {SR_EXPECTED} Hz, got {fin.samplerate} Hz",
                    "score": 0,
                    "tampered_regions": [],
                    "total_tampered_duration": 0
                }
            
            total_samples = len(fin)
            chunk_size = int(CHUNK_SEC * SR_EXPECTED)
            
            total_frames = 0
            total_pos_frames = 0
            gaps = []
            
            cursor_samples = 0
            
            while True:
                audio = fin.read(frames=chunk_size, dtype="float32", always_2d=False)
                if not isinstance(audio, np.ndarray) or audio.size == 0:
                    break
                
                t = to_tensor_mono(audio)
                
                with torch.no_grad():
                    frame_probs, msg_probs = detector(t, SR_EXPECTED)
                wm_probs = frame_probs[:, 1, :].squeeze(0).cpu().numpy()
                
                total_frames += wm_probs.shape[0]
                total_pos_frames += np.sum(wm_probs >= FRAME_THRESH)
                
                below = wm_probs < FRAME_THRESH
                if np.any(below):
                    idx = np.flatnonzero(np.diff(np.concatenate(([0], below.view(np.int8), [0]))))
                    runs = list(zip(idx[0::2], idx[1::2]))
                    for s, e in runs:
                        dur_sec = (e - s) / SR_EXPECTED
                        if dur_sec >= MIN_GAP_SEC:
                            start_sec = (cursor_samples + s) / SR_EXPECTED
                            end_sec = (cursor_samples + e) / SR_EXPECTED
                            gaps.append({
                                "start_time": round(start_sec, 2),
                                "end_time": round(end_sec, 2),
                                "duration": round(dur_sec, 2)
                            })
                
                cursor_samples += audio.shape[0]
            
            overall_score = 100.0 * (total_pos_frames / max(1, total_frames))
            total_tampered = sum(g["duration"] for g in gaps)
            
            return {
                "status": "TAMPERED" if gaps else "AUTHENTIC",
                "score": round(overall_score, 2),
                "tampered_regions": gaps,
                "total_tampered_duration": round(total_tampered, 2),
                "duration_minutes": round(total_samples / SR_EXPECTED / 60.0, 2)
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "score": 0,
            "tampered_regions": [],
            "total_tampered_duration": 0
        }

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Select Audio File")
    
    audio_files = {
        "Official Recording (Signed)": "hearing.signed.wav",
        "Tampered Version": "hearing.tampered.wav",
        "Watermarked (No C2PA)": "hearing.wm.wav",
        "Raw Original": "hearing.raw.wav"
    }
    
    selected_file_label = st.selectbox(
        "Choose an audio file to verify:",
        list(audio_files.keys())
    )
    
    selected_file = audio_files[selected_file_label]
    file_path = Path("/Users/aunabbas/Documents/media-provenance-mvp") / selected_file
    
    if file_path.exists():
        st.success(f"✓ File loaded: {selected_file}")
        
        st.subheader("🔊 Audio Player")
        st.audio(str(file_path))
        
        file_stats = os.stat(file_path)
        st.info(f"File size: {file_stats.st_size / (1024*1024):.2f} MB")
    else:
        st.error(f"File not found: {selected_file}")

with col2:
    st.subheader("🔍 Verification Tools")
    
    if file_path.exists():
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("📜 View Content Credentials", use_container_width=True):
                with st.spinner("Checking C2PA credentials..."):
                    c2pa_output = run_c2patool(str(file_path))
                    
                    st.subheader("C2PA Verification Results")
                    
                    if "No claim or manifest found" in c2pa_output or "No manifests found" in c2pa_output or "No C2PA Manifests" in c2pa_output:
                        st.error("❌ No Content Credentials found")
                        st.code(c2pa_output, language="text")
                    elif "Validated" in c2pa_output or "Successfully validated" in c2pa_output:
                        st.success("✅ Valid Content Credentials found")
                        st.code(c2pa_output, language="text")
                    else:
                        st.warning("⚠️ Content Credentials present but validation unclear")
                        st.code(c2pa_output, language="text")
        
        with col2b:
            if st.button("🔐 Check Watermark", use_container_width=True):
                with st.spinner("Analyzing AudioSeal watermark..."):
                    result = check_watermark(str(file_path))
                    
                    st.subheader("Watermark Analysis Results")
                    
                    if result["status"] == "ERROR":
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    else:
                        if result["status"] == "AUTHENTIC":
                            st.success(f"✅ AUTHENTIC - Watermark Score: {result['score']}%")
                        else:
                            st.error(f"⚠️ TAMPERED - Watermark Score: {result['score']}%")
                        
                        st.metric("Overall Watermark Score", f"{result['score']}%")
                        st.metric("Audio Duration", f"{result.get('duration_minutes', 0)} minutes")
                        
                        if result["tampered_regions"]:
                            st.error(f"🚨 Found {len(result['tampered_regions'])} tampered region(s)")
                            st.metric("Total Tampered Duration", f"{result['total_tampered_duration']} seconds")
                            
                            st.subheader("Tampered Regions Detail:")
                            for i, region in enumerate(result["tampered_regions"], 1):
                                st.warning(
                                    f"Region {i}: {region['start_time']}s → {region['end_time']}s "
                                    f"(Duration: {region['duration']}s)"
                                )
                        else:
                            st.success("No tampered regions detected")

st.divider()

with st.expander("ℹ️ About This System"):
    st.markdown("""
    ### How It Works
    
    This system uses two complementary technologies to verify audio authenticity:
    
    1. **C2PA (Content Credentials)**: Cryptographically signed metadata that proves the origin and authenticity of the file
    2. **AudioSeal Watermarking**: Imperceptible audio watermarks that can detect tampering even in modified copies
    
    ### Verification Process
    
    - **Official recordings** should have both valid C2PA credentials AND no watermark gaps
    - **Tampered recordings** will typically fail C2PA verification and/or show gaps in the watermark
    - **Watermark gaps ≥ 1 second** indicate potential tampering or manipulation
    
    ### File Types in Demo
    
    - **Official Recording (Signed)**: Contains both C2PA signature and AudioSeal watermark
    - **Tampered Version**: Modified version with ~10 second insertion (fails both checks)
    - **Watermarked (No C2PA)**: Has watermark but no C2PA signature
    - **Raw Original**: Original file before any processing
    """)

with st.expander("🔧 Technical Details"):
    st.code("""
    Configuration:
    - Sample Rate: 16000 Hz
    - Chunk Size: 10 seconds
    - Frame Threshold: 0.5
    - Minimum Gap Detection: 1.0 seconds
    - Watermark Model: audioseal_detector_16bits
    """, language="python")