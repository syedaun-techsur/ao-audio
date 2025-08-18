import streamlit as st
import subprocess
import json
import numpy as np
import soundfile as sf
import torch
from audioseal import AudioSeal
import os
from pathlib import Path
import tempfile

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

# Initialize session state for file path
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

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

def decode_message_to_id(message_tensor):
    """Convert the decoded message tensor back to manifest ID"""
    if message_tensor is None:
        return None
    
    message_np = message_tensor.cpu().numpy().flatten()
    manifest_id = 0
    for i in range(16):
        if i < len(message_np):
            bit = int(message_np[i] > 0.5)
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
            "location": "US District Court, Southern District"
        },
        54321: {
            "creator": "Official Court Recording",
            "date": "2024-02-20",
            "court_id": "USDC-2024-047",
            "title": "Preliminary Hearing - State v. Johnson",
            "recorder": "Court Reporter John Doe",
            "location": "US District Court, Northern District"
        }
    }
    return manifests_db.get(manifest_id, None)

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

def check_durable_credentials(file_path):
    """Check durable content credentials by recovering manifest from watermark"""
    try:
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        
        with sf.SoundFile(file_path, "r") as fin:
            if fin.samplerate != SR_EXPECTED:
                return {
                    "status": "ERROR",
                    "error": f"Expected {SR_EXPECTED} Hz, got {fin.samplerate} Hz"
                }
            
            # Read first 30 seconds for message detection
            max_samples = min(len(fin), 30 * SR_EXPECTED)
            audio = fin.read(frames=max_samples, dtype="float32", always_2d=False)
            
            if audio.size == 0:
                return {"status": "ERROR", "error": "Empty audio file"}
            
            t = to_tensor_mono(audio)
            
            # Detect watermark and extract message
            with torch.no_grad():
                result, message = detector.detect_watermark(t, SR_EXPECTED)
                frame_probs, _ = detector(t, SR_EXPECTED)
            
            detection_prob = float(result)
            wm_probs = frame_probs[:, 1, :].squeeze(0).cpu().numpy()
            
            # Check for tampering (quick scan)
            fin.seek(0)
            total_samples = len(fin)
            chunk_size = int(CHUNK_SEC * SR_EXPECTED)
            gaps = []
            cursor_samples = 0
            
            while True:
                audio_chunk = fin.read(frames=chunk_size, dtype="float32", always_2d=False)
                if not isinstance(audio_chunk, np.ndarray) or audio_chunk.size == 0:
                    break
                
                t_chunk = to_tensor_mono(audio_chunk)
                with torch.no_grad():
                    fp, _ = detector(t_chunk, SR_EXPECTED)
                wp = fp[:, 1, :].squeeze(0).cpu().numpy()
                
                below = wp < FRAME_THRESH
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
                cursor_samples += audio_chunk.shape[0]
            
            if detection_prob < 0.5:
                return {
                    "status": "NO_WATERMARK",
                    "watermark_detected": False,
                    "manifest_recovered": False,
                    "manifest": None,
                    "tampered_regions": []
                }
            
            # Decode manifest ID from message
            manifest_id = decode_message_to_id(message) if message is not None else None
            manifest = get_mock_manifest_from_id(manifest_id) if manifest_id else None
            
            # Determine integrity percentage
            total_tampered = sum(g["duration"] for g in gaps)
            total_duration = total_samples / SR_EXPECTED
            integrity_pct = 100 * (1 - total_tampered / max(1, total_duration))
            
            return {
                "status": "RECOVERED" if manifest else "WATERMARK_ONLY",
                "watermark_detected": True,
                "watermark_integrity": round(integrity_pct, 1),
                "manifest_recovered": manifest is not None,
                "manifest_id": manifest_id,
                "manifest": manifest,
                "tampered_regions": gaps,
                "total_tampered_duration": round(total_tampered, 2)
            }
            
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e)
        }

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file to verify:",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload a court hearing recording to verify its authenticity"
    )
    
    if uploaded_file is not None:
        # Only create a new temp file if the uploaded file has changed
        file_bytes = uploaded_file.getvalue()
        file_key = f"{uploaded_file.name}_{len(file_bytes)}"
        
        if 'current_file_key' not in st.session_state or st.session_state.current_file_key != file_key:
            # Clean up old temp file if it exists
            if st.session_state.temp_file_path and Path(st.session_state.temp_file_path).exists():
                try:
                    os.unlink(st.session_state.temp_file_path)
                except:
                    pass
            
            # Create new temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(file_bytes)
                st.session_state.temp_file_path = tmp_file.name
                st.session_state.current_file_key = file_key
        
        file_path = Path(st.session_state.temp_file_path)
        
        st.success(f"✓ File loaded: {uploaded_file.name}")
        
        st.subheader("🔊 Audio Player")
        st.audio(uploaded_file)
    else:
        file_path = None
        st.info("Please upload an audio file to verify")

with col2:
    st.subheader("🔍 Verification Tools")
    
    if file_path is not None and file_path.exists():
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
                            st.success("✅ AUTHENTIC")
                        else:
                            st.error("⚠️ TAMPERED")
                        
                        if result["tampered_regions"]:
                            st.error(f"🚨 Found {len(result['tampered_regions'])} tampered region(s)")
                            st.metric("Total Tampered Duration", f"{result['total_tampered_duration']} seconds")
                            
                            st.subheader("Tampered Regions Detail:")
                            for i, region in enumerate(result["tampered_regions"], 1):
                                start_min = int(region['start_time'] // 60)
                                start_sec = int(region['start_time'] % 60)
                                end_min = int(region['end_time'] // 60)
                                end_sec = int(region['end_time'] % 60)
                                st.warning(
                                    f"Region {i}: {start_min:02d}:{start_sec:02d} → {end_min:02d}:{end_sec:02d}"
                                )
                        else:
                            st.success("No tampered regions detected")
        
        # Add the Durable Credentials button as a full-width button
        if st.button("🔗 Check Durable Credentials", use_container_width=True, help="Recover C2PA manifest from watermark even if metadata is stripped"):
            with st.spinner("Checking durable credentials..."):
                result = check_durable_credentials(str(file_path))
                
                st.subheader("🔗 Durable Credentials Recovery")
                
                if result["status"] == "ERROR":
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                elif result["status"] == "NO_WATERMARK":
                    st.error("❌ NO WATERMARK DETECTED")
                    st.markdown("Cannot recover credentials - no watermark found in audio")
                
                elif result["status"] == "WATERMARK_ONLY":
                    st.warning("⚠️ WATERMARK DETECTED BUT NO MANIFEST")
                    st.markdown(f"Watermark Integrity: {result['watermark_integrity']}%")
                    st.info("Watermark present but no manifest ID found (may be legacy watermark)")
                
                elif result["status"] == "RECOVERED":
                    if result["tampered_regions"]:
                        # Prominent tampering warning at top
                        st.error("⚠️ TAMPERING DETECTED - CREDENTIALS RECOVERED")
                        
                        # Show tampering details prominently
                        st.markdown("### ⚠️ Tampered Regions Found:")
                        for i, region in enumerate(result["tampered_regions"], 1):
                            start_min = int(region['start_time'] // 60)
                            start_sec = int(region['start_time'] % 60)
                            end_min = int(region['end_time'] // 60)
                            end_sec = int(region['end_time'] % 60)
                            duration = region['duration']
                            st.warning(f"Region {i}: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} ({duration} seconds)")
                        
                        st.metric("Total Tampered Duration", f"{result['total_tampered_duration']} seconds")
                        
                        # Then show recovered credentials
                        st.success("✅ Original C2PA Credentials Successfully Recovered:")
                        manifest = result["manifest"]
                        st.markdown(f"""
                        - **Creator:** {manifest['creator']}
                        - **Date:** {manifest['date']}
                        - **Court ID:** {manifest['court_id']}
                        - **Title:** {manifest['title']}
                        - **Recorder:** {manifest['recorder']}
                        - **Location:** {manifest['location']}
                        """)
                    
                    else:
                        st.success("✅ DURABLE CREDENTIALS FULLY INTACT")
                        
                        st.info("No tampering detected - Full integrity maintained")
                        
                        st.markdown("### Recovered C2PA Credentials:")
                        manifest = result["manifest"]
                        st.markdown(f"""
                        - **Creator:** {manifest['creator']}  
                        - **Date:** {manifest['date']}  
                        - **Court ID:** {manifest['court_id']}  
                        - **Title:** {manifest['title']}  
                        - **Recorder:** {manifest['recorder']}  
                        - **Location:** {manifest['location']}
                        """)

st.divider()

with st.expander("📖 How to Use This System"):
    st.markdown("""
    ### Getting Started
    
    **Step 1: Upload Your Audio File**
    - Click "Browse files" or drag and drop your audio file
    - Supported formats: WAV, MP3, M4A, FLAC, OGG
    - You'll see the audio player once uploaded
    
    **Step 2: Choose Your Verification Method**
    
    You have three verification options:
    
    1. **📜 View Content Credentials**
       - Click this to check the digital seal
       - Shows who created the file and when
       - Best for: Verifying official documents
    
    2. **🔐 Check Watermark**
       - Click this to scan for hidden watermarks
       - Detects any tampering in the audio
       - Best for: Finding edited sections
    
    3. **🔗 Check Durable Credentials**
       - Click this for our most advanced check
       - Recovers information even from damaged files
       - Best for: Files from social media or unknown sources
    
    **Step 3: Understand Your Results**
    
    - **Green messages** = Good news, file is authentic
    - **Yellow warnings** = Caution, some issues detected
    - **Red alerts** = Problems found, see details
    
    **Pro Tip:** For the most thorough verification, run all three checks!
    """)

with st.expander("🏗️ Durable Content Credentials Architecture"):
    st.markdown("""
    ### How Durable Content Credentials Work
    
    This diagram shows the breakthrough technology that allows credential recovery even when traditional C2PA metadata is stripped:
    """)
    
    try:
        st.image("architecture.png", caption="Durable Content Credentials System Architecture", use_container_width=True)
    except:
        st.error("Architecture diagram not found. Please ensure 'architecture.png' is in the project directory.")
        st.markdown("""
        **Durable Credentials Process:**
        - Manifest ID embedded in watermark during creation
        - ID links to credential database containing court information
        - Recovery possible even when C2PA signatures are lost
        - Shows both original source AND tampering detection
        """)

with st.expander("ℹ️ About This System"):
    st.markdown("""
    ## Welcome to the Court Audio Verification System
    
    This cutting-edge system ensures the authenticity of court audio recordings using **three layers of protection**. 
    Think of it as a digital fingerprint, invisible watermark, and recovery system all working together to protect the truth.
    
    ---
    
    ### **Layer 1: Content Credentials (C2PA)**
    
    **What it does:** Acts like a digital seal on the audio file
    
    **How it helps you:**
    - Proves WHO created the recording (court reporter, judge, etc.)
    - Shows WHEN it was created (date and time)
    - Confirms WHERE it came from (which court)
    - Ensures the file hasn't been modified since signing
    
    **Limitation:** This seal can be lost when files are uploaded to social media or messaging apps
    
    ---
    
    ### **Layer 2: Watermark Detection (AudioSeal)**
    
    **What it does:** Embeds an invisible pattern throughout the entire audio
    
    **How it helps you:**
    - Detects if ANY part of the audio has been altered
    - Shows EXACTLY where tampering occurred (down to the second)
    - Works even after file conversion or compression
    
    **Think of it like:** Invisible ink that reveals tampering when checked
    
    ---
    
    ### **Layer 3: Durable Credentials (NEW!)**
    
    **What it does:** Recovers the original credentials even when traditional methods fail
    
    **How it helps you:**
    - Recovers court information even after social media uploads
    - Works when the digital seal (C2PA) has been stripped
    - Shows BOTH the original source AND any tampering
    - Provides evidence trail even for modified files
    
    **The breakthrough:** Even if someone tampers with the audio and removes the digital seal, 
    we can still prove it came from an official court recording and show exactly what was changed.
    
    ---
    
    ### Real-World Scenarios
    
    **Scenario 1: Official Recording**
    - All three checks pass
    - File is 100% authentic and unmodified
    - Full chain of custody maintained
    
    **Scenario 2: Social Media Upload**
    - C2PA seal lost (stripped by platform)
    - Watermark intact
    - Durable credentials recoverable
    - Original source verified despite platform processing
    
    **Scenario 3: Tampered Evidence**
    - C2PA seal broken
    - Watermark shows gaps
    - Durable credentials still recovers original info
    - Shows exact timestamps of tampering
    
    ---
    
    ### Why This Matters
    
    In today's world of deepfakes and manipulated media, this system provides:
    - **Legal certainty** for court proceedings
    - **Evidence integrity** for investigations
    - **Public trust** in official recordings
    - **Tamper evidence** that stands up in court
    """)
