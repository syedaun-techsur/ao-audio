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
from datetime import datetime
import uuid

SR_EXPECTED = 16000
CHUNK_SEC = 10
FRAME_THRESH = 0.5
MIN_GAP_SEC = 1.0

st.set_page_config(
    page_title="Court Audio Authentication System",
    page_icon="⚖️",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("⚖️ Court Audio System")
mode = st.sidebar.selectbox(
    "Choose Mode:",
    ["🔍 Verify Audio", "📤 Embed Credentials", "🔥 Add Tampering"],
    help="Select whether to verify existing audio, create authenticated audio, or add tampering for testing"
)

if "🔍" in mode:
    st.title("⚖️ Court Audio Verification System")
    st.markdown("### Verify authenticity of court hearing recordings using C2PA and AudioSeal")
elif "📤" in mode:
    st.title("📤 Court Audio Authentication System")
    st.markdown("### Create authenticated court recordings with C2PA and AudioSeal watermarks")
else:
    st.title("⚙️ Audio Tampering Tool")
    st.markdown("### Add controlled tampering to audio files for testing verification systems")

# Initialize session state for file paths
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'embed_temp_file_path' not in st.session_state:
    st.session_state.embed_temp_file_path = None

# Data store file path
MANIFEST_STORE_PATH = "manifest_store.json"

def load_manifest_store():
    """Load manifest store from JSON file"""
    try:
        if os.path.exists(MANIFEST_STORE_PATH):
            with open(MANIFEST_STORE_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading manifest store: {e}")
        return {}

def save_manifest_store(store):
    """Save manifest store to JSON file"""
    try:
        with open(MANIFEST_STORE_PATH, 'w') as f:
            json.dump(store, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving manifest store: {e}")
        return False

def add_manifest_to_store(manifest_id, manifest_data):
    """Add a new manifest to the store"""
    store = load_manifest_store()
    store[str(manifest_id)] = manifest_data
    return save_manifest_store(store)

def add_tampering_to_audio(input_path, output_path, noise_duration=10, noise_level=0.1):
    """Add noise tampering to the end of audio file"""
    try:
        with sf.SoundFile(input_path, "r") as fin:
            # Read original audio
            audio_data = fin.read(dtype="float32", always_2d=False)
            sample_rate = fin.samplerate
            
            # Convert to mono if stereo
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
            
            # Generate noise with specified duration and level
            noise_samples = int(noise_duration * sample_rate)
            noise = np.random.normal(0, noise_level, noise_samples).astype(np.float32)
            
            # Concatenate original audio with noise
            tampered_audio = np.concatenate([audio_data, noise])
            
            # Write tampered audio
            with sf.SoundFile(output_path, "w", samplerate=sample_rate, channels=1, subtype="PCM_16") as fout:
                fout.write(tampered_audio)
            
            return True, f"Added {noise_duration} seconds of noise tampering (level: {noise_level})"
    except Exception as e:
        return False, str(e)

def embed_watermark_with_id(input_path, output_path, manifest_id):
    """Embed AudioSeal watermark with manifest ID"""
    try:
        torch.set_num_threads(max(1, torch.get_num_threads() // 2))
        gen = AudioSeal.load_generator("audioseal_wm_16bits")
        
        # Create message tensor for the manifest ID
        message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
        for i in range(16):
            bit = (manifest_id >> i) & 1
            message[0, i] = bit
        
        with sf.SoundFile(input_path, "r") as fin:
            if fin.samplerate != SR_EXPECTED:
                return False, f"Expected {SR_EXPECTED} Hz, got {fin.samplerate}"
            
            n_samples = len(fin)
            chunk_size = 30 * SR_EXPECTED  # 30 second chunks
            
            with sf.SoundFile(output_path, "w", samplerate=SR_EXPECTED, channels=1, subtype="PCM_16") as fout:
                chunk_idx = 0
                while True:
                    audio = fin.read(frames=chunk_size, dtype="float32", always_2d=False)
                    if not isinstance(audio, np.ndarray) or audio.size == 0:
                        break
                    
                    # Convert to tensor
                    if audio.ndim == 2:
                        audio = np.mean(audio, axis=1)
                    t = torch.from_numpy(audio).to(torch.float32).unsqueeze(0).unsqueeze(0)
                    
                    # Generate watermark using correct AudioSeal API
                    with torch.no_grad():
                        watermark = gen.get_watermark(t, SR_EXPECTED, message=message)
                        watermarked = t + watermark
                    
                    # Convert back and write
                    watermarked_np = watermarked.squeeze().cpu().numpy()
                    fout.write(watermarked_np)
                    chunk_idx += 1
        
        return True, "Watermark embedded successfully"
    except Exception as e:
        return False, str(e)

def create_c2pa_manifest(metadata):
    """Create C2PA manifest file from metadata"""
    manifest = {
        "ta_url": "http://timestamp.digicert.com",
        "claim_generator": metadata.get('claim_generator', 'Court Audio Authentication System/1.0'),
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "when": metadata.get('date_created', datetime.now().isoformat())
                        }
                    ]
                }
            },
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "CreativeWork",
                    "name": metadata.get('title', ''),
                    "author": [
                        {
                            "@type": "Person",
                            "name": metadata.get('author', '')
                        }
                    ],
                    "dateCreated": metadata.get('date_created', datetime.now().isoformat()),
                    "locationCreated": {
                        "@type": "Place",
                        "name": metadata.get('location', '')
                    },
                    "identifier": metadata.get('court_id', '')
                }
            }
        ]
    }
    return manifest

def run_c2patool(file_path, detailed=False):
    """Run c2patool on a file and return the output"""
    try:
        cmd = ["c2patool", file_path]
        if detailed:
            cmd.append("--detailed")
        else:
            cmd.append("--info")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error running c2patool: {str(e)}"

def parse_c2pa_output(c2pa_output):
    """Parse c2patool detailed JSON output to extract metadata"""
    try:
        # Try to parse as JSON first (for detailed output)
        if c2pa_output.strip().startswith('{'):
            data = json.loads(c2pa_output)
            metadata = {}
            
            # Extract from manifests
            if 'manifests' in data and data['manifests']:
                # Get the active manifest
                active_manifest_id = data.get('active_manifest')
                if active_manifest_id and active_manifest_id in data['manifests']:
                    manifest = data['manifests'][active_manifest_id]
                else:
                    # Fallback to first manifest
                    manifest = list(data['manifests'].values())[0]
                
                # Extract claim data
                if 'claim' in manifest:
                    claim = manifest['claim']
                    
                    # Basic info
                    metadata['title'] = claim.get('dc:title', 'Unknown')
                    metadata['format'] = claim.get('dc:format', 'Unknown')
                    metadata['instance_id'] = claim.get('instanceID', 'Unknown')
                    metadata['claim_generator'] = claim.get('claim_generator', 'Unknown')
                    
                    # Claim generator info
                    if 'claim_generator_info' in claim:
                        claim_info = claim['claim_generator_info']
                        if isinstance(claim_info, list) and claim_info:
                            info = claim_info[0]
                            metadata['generator_name'] = info.get('name', 'Unknown')
                            metadata['generator_version'] = info.get('version', 'Unknown')
                
                # Extract assertion data
                if 'assertion_store' in manifest:
                    assertions = manifest['assertion_store']
                    
                    # Actions
                    if 'c2pa.actions.v2' in assertions:
                        actions_data = assertions['c2pa.actions.v2']
                        if 'actions' in actions_data and actions_data['actions']:
                            action = actions_data['actions'][0]
                            metadata['action'] = action.get('action', 'Unknown').replace('c2pa.', '')
                            metadata['when'] = action.get('when', 'Unknown')
                    
                    # Schema.org CreativeWork
                    if 'stds.schema-org.CreativeWork' in assertions:
                        creative_work = assertions['stds.schema-org.CreativeWork']
                        metadata['content_title'] = creative_work.get('name', 'Unknown')
                        metadata['identifier'] = creative_work.get('identifier', 'Unknown')
                        metadata['date_created'] = creative_work.get('dateCreated', 'Unknown')
                        
                        # Author info
                        if 'author' in creative_work and creative_work['author']:
                            if isinstance(creative_work['author'], list):
                                author = creative_work['author'][0]
                            else:
                                author = creative_work['author']
                            metadata['author'] = author.get('name', 'Unknown')
                        
                        # Location
                        if 'locationCreated' in creative_work:
                            location = creative_work['locationCreated']
                            metadata['location'] = location.get('name', 'Unknown')
                
                # Extract signature info
                if 'signature' in manifest:
                    sig_info = manifest['signature']
                    metadata['signature_algorithm'] = sig_info.get('alg', 'Unknown')
                    metadata['issuer'] = sig_info.get('issuer', 'Unknown')
                    metadata['signature_time'] = sig_info.get('time', 'Unknown')
            
            # Validation state
            metadata['validation_state'] = data.get('validation_state', 'Unknown')
            
            return metadata
            
        else:
            # Parse text output for --info mode
            lines = c2pa_output.split('\n')
            metadata = {}
            
            for line in lines:
                line = line.strip()
                if 'Information for' in line:
                    # Extract filename
                    filename = line.replace('Information for ', '').strip()
                    metadata['filename'] = filename
                elif 'Manifest store size' in line:
                    # Extract manifest info
                    metadata['manifest_info'] = line
                elif line == 'Validated':
                    metadata['validation_state'] = 'Valid'
                elif 'manifest' in line.lower():
                    metadata['manifest_count'] = line
            
            return metadata
            
    except Exception as e:
        return {}

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

def get_manifest_from_id(manifest_id):
    """Retrieve manifest from watermark ID using local store"""
    store = load_manifest_store()
    return store.get(str(manifest_id), None)

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
            manifest = get_manifest_from_id(manifest_id) if manifest_id else None
            
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

# Main content area - conditional based on mode
if "🔍" in mode:
    # VERIFICATION MODE
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
                        # First try detailed output
                        c2pa_detailed = run_c2patool(str(file_path), detailed=True)
                        c2pa_info = run_c2patool(str(file_path), detailed=False)
                        
                        st.subheader("📜 C2PA Verification Results")
                        
                        if "No claim or manifest found" in c2pa_info or "No manifests found" in c2pa_info or "No C2PA Manifests" in c2pa_info:
                            st.error("❌ No Content Credentials found")
                            st.info("This file does not contain C2PA metadata. Try 'Check Durable Credentials' to recover information from the watermark.")
                            
                            with st.expander("View Raw Output"):
                                st.code(c2pa_info, language="text")
                        
                        elif "Validated" in c2pa_info or "Successfully validated" in c2pa_info:
                            st.success("✅ Valid Content Credentials Found")
                            
                            # Parse detailed metadata
                            metadata = parse_c2pa_output(c2pa_detailed)
                            
                            if metadata and len([k for k, v in metadata.items() if v != 'Unknown']) > 0:
                                st.markdown("### 📋 Content Metadata:")
                                
                                # Core content information
                                if metadata.get('content_title', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Content Title:** {metadata['content_title']}")
                                elif metadata.get('title', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **File Title:** {metadata['title']}")
                                
                                if metadata.get('identifier', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Court Case ID:** {metadata['identifier']}")
                                
                                if metadata.get('author', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Author:** {metadata['author']}")
                                
                                if metadata.get('location', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Location:** {metadata['location']}")
                                
                                if metadata.get('date_created', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Date Created:** {metadata['date_created']}")
                                elif metadata.get('when', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **When:** {metadata['when']}")
                                
                                if metadata.get('action', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Action:** {metadata['action'].title()}")
                                
                                if metadata.get('format', 'Unknown') != 'Unknown':
                                    st.markdown(f"- **Format:** {metadata['format']}")
                            else:
                                st.info("✅ Credentials validated but no detailed metadata could be parsed.")
                            
                            with st.expander("View Raw C2PA Output"):
                                st.code(c2pa_detailed if c2pa_detailed.strip().startswith('{') else c2pa_info, language="json" if c2pa_detailed.strip().startswith('{') else "text")
                        
                        else:
                            st.warning("⚠️ Content Credentials present but validation unclear")
                            
                            # Try to parse metadata even if validation is unclear
                            metadata = parse_c2pa_output(c2pa_detailed)
                            
                            if metadata and len([k for k, v in metadata.items() if v != 'Unknown']) > 0:
                                st.markdown("### 📋 Extracted Metadata:")
                                
                                # Display available metadata
                                display_fields = [
                                    ('content_title', 'Content Title'),
                                    ('title', 'File Title'),
                                    ('identifier', 'Court Case ID'),
                                    ('author', 'Author'),
                                    ('location', 'Location'),
                                    ('date_created', 'Date Created'),
                                    ('when', 'When'),
                                    ('action', 'Action'),
                                    ('format', 'Format'),
                                    ('validation_state', 'Validation State')
                                ]
                                
                                for key, display_name in display_fields:
                                    if metadata.get(key, 'Unknown') != 'Unknown':
                                        value = metadata[key]
                                        if key == 'action':
                                            value = value.title()
                                        st.markdown(f"- **{display_name}:** {value}")
                            
                            with st.expander("View Raw C2PA Output"):
                                st.code(c2pa_detailed if c2pa_detailed.strip().startswith('{') else c2pa_info, language="json" if c2pa_detailed.strip().startswith('{') else "text")
            
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
                            st.success("✅ Original Credentials Successfully Recovered:")
                            manifest = result["manifest"]
                            
                            # Display all manifest fields dynamically
                            for key, value in manifest.items():
                                # Format key for display (convert snake_case to Title Case)
                                display_key = key.replace('_', ' ').title()
                                st.markdown(f"- **{display_key}:** {value}")
                        
                        else:
                            st.success("✅ DURABLE CREDENTIALS FULLY INTACT")
                            
                            st.info("No tampering detected - Full integrity maintained")
                            
                            st.markdown("### Recovered Credentials:")
                            manifest = result["manifest"]
                            
                            # Display all manifest fields dynamically
                            for key, value in manifest.items():
                                # Format key for display (convert snake_case to Title Case)
                                display_key = key.replace('_', ' ').title()
                                st.markdown(f"- **{display_key}:** {value}")
    
elif "📤" in mode:
    # EMBEDDING MODE
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Original Court Audio")
        
        embed_uploaded_file = st.file_uploader(
            "Choose an original court audio file to authenticate:",
            type=["wav"],
            help="Upload a WAV file to embed with C2PA credentials and watermark",
            key="embed_uploader"
        )
        
        if embed_uploaded_file is not None:
            # Handle embed temp file
            file_bytes = embed_uploaded_file.getvalue()
            file_key = f"{embed_uploaded_file.name}_{len(file_bytes)}"
            
            if 'embed_current_file_key' not in st.session_state or st.session_state.embed_current_file_key != file_key:
                # Clean up old temp file if it exists
                if st.session_state.embed_temp_file_path and Path(st.session_state.embed_temp_file_path).exists():
                    try:
                        os.unlink(st.session_state.embed_temp_file_path)
                    except:
                        pass
                
                # Create new temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(file_bytes)
                    st.session_state.embed_temp_file_path = tmp_file.name
                    st.session_state.embed_current_file_key = file_key
            
            embed_file_path = Path(st.session_state.embed_temp_file_path)
            
            st.success(f"✓ File loaded: {embed_uploaded_file.name}")
            
            # Check audio format
            try:
                with sf.SoundFile(embed_file_path, "r") as f:
                    if f.samplerate != SR_EXPECTED:
                        st.warning(f"⚠️ Audio is {f.samplerate} Hz. For best results, use 16kHz mono WAV files.")
                    st.info(f"Duration: {len(f) / f.samplerate:.1f} seconds, Sample rate: {f.samplerate} Hz")
            except:
                st.error("Error reading audio file. Please ensure it's a valid WAV file.")
                embed_file_path = None
            
            st.subheader("🔊 Audio Preview")
            st.audio(embed_uploaded_file)
        else:
            embed_file_path = None
            st.info("Please upload a WAV court audio file to authenticate")
    
    with col2:
        st.subheader("📝 Content Credentials Form")
        
        with st.form("metadata_form"):
            st.markdown("**Court Information**")
            court_id = st.text_input("Court Case ID*", placeholder="e.g., USDC-2024-001")
            title = st.text_input("Case Title*", placeholder="e.g., Hearing on Case No. 2024-CV-00123")
            location = st.text_input("Court Location*", placeholder="e.g., US District Court, Southern District")
            
            st.markdown("**Recording Information**")
            recorder = st.text_input("Court Reporter*", placeholder="e.g., Court Reporter Jane Smith")
            creator = st.text_input("Creator", value="Official Court Recording", placeholder="Recording system or entity")
            
            st.markdown("**Timestamp**")
            date_created = st.date_input("Date Created", value=datetime.now().date())
            time_created = st.time_input("Time Created", value=datetime.now().time())
            
            st.markdown("**Output Settings**")
            output_filename = st.text_input("Output Filename*", placeholder="e.g., court1", help="Base name for the output file (without extension)")
            
            if output_filename:
                st.info(f"💾 Output will be: **{output_filename}.signed.wav** (downloaded to your local PC)")
                st.caption("Note: If C2PA signing fails, you'll get **{}.watermarked.wav** instead".format(output_filename))
            
            st.markdown("**Advanced Settings**")
            claim_generator = st.text_input("Claim Generator", value="Court Audio Authentication System/1.0")
            manifest_id = st.number_input("Manifest ID (0-65535)", min_value=0, max_value=65535, value=int(datetime.now().timestamp()) % 65536, help="Unique 16-bit identifier for this recording")
            
            # Custom metadata fields
            st.markdown("**Additional Metadata (Optional)**")
            custom_fields = st.text_area("Custom JSON Fields", placeholder='{"judge": "Hon. John Doe", "case_type": "Civil", "proceeding_type": "Motion Hearing"}', help="Additional court metadata as JSON object")
            
            submitted = st.form_submit_button("Create Authenticated Audio", type="primary", use_container_width=True)
            
            if submitted:
                if not all([court_id, title, location, recorder, output_filename]):
                    st.error("⚠️ Please fill in all required fields marked with *")
                elif embed_file_path is None:
                    st.error("⚠️ Please upload an audio file first")
                else:
                    # Create datetime string
                    datetime_str = datetime.combine(date_created, time_created).isoformat()
                    
                    # Prepare metadata
                    metadata = {
                        "court_id": court_id,
                        "title": title,
                        "location": location,
                        "recorder": recorder,
                        "creator": creator,
                        "date_created": datetime_str,
                        "claim_generator": claim_generator,
                        "date": date_created.isoformat()
                    }
                    
                    # Add custom fields if provided
                    if custom_fields.strip():
                        try:
                            custom_data = json.loads(custom_fields)
                            metadata.update(custom_data)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON in custom fields. Please check your syntax.")
                            st.stop()
                    
                    # Store form data in session state for processing outside form
                    st.session_state.process_audio = {
                        'embed_file_path': embed_file_path,
                        'metadata': metadata,
                        'manifest_id': manifest_id,
                        'output_filename': output_filename
                    }
                    st.success("✓ Audio processing queued - see results below!")
                    st.rerun()
        
        # Process audio outside of form if queued
        if 'process_audio' in st.session_state and st.session_state.process_audio:
            process_data = st.session_state.process_audio
            embed_file_path = process_data['embed_file_path']
            metadata = process_data['metadata']
            manifest_id = process_data['manifest_id']
            output_filename = process_data['output_filename']
            
            st.markdown("---")
            st.subheader("🔄 Processing Your Audio...")
            
            # Option to cancel processing
            col_process, col_cancel = st.columns([3, 1])
            with col_cancel:
                if st.button("Cancel Processing", type="secondary"):
                    del st.session_state.process_audio
                    st.rerun()
            
            with st.spinner("Creating authenticated audio..."):
                try:
                    # Step 1: Embed watermark
                    watermarked_path = embed_file_path.with_suffix('.watermarked.wav')
                    success, message = embed_watermark_with_id(str(embed_file_path), str(watermarked_path), manifest_id)
                    
                    if not success:
                        st.error(f"Watermark embedding failed: {message}")
                        # Clear processing state
                        del st.session_state.process_audio
                        st.stop()
                    
                    st.success("✓ Step 1: Watermark embedded successfully")
                    
                    # Step 2: Create C2PA manifest
                    manifest = create_c2pa_manifest(metadata)
                    manifest_path = embed_file_path.with_suffix('.manifest.json')
                    with open(manifest_path, 'w') as f:
                        json.dump(manifest, f, indent=2)
                    
                    st.success("✓ Step 2: C2PA manifest created")
                    
                    # Step 3: Sign with C2PA
                    signed_path = embed_file_path.with_suffix('.signed.wav')
                    result = subprocess.run(
                        ["c2patool", str(watermarked_path), "-m", str(manifest_path), "-o", str(signed_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        st.error(f"C2PA signing failed: {result.stderr}")
                        st.info("Note: This requires c2patool to be installed and available in PATH")
                        
                        # Offer watermarked file as fallback
                        st.info("Providing watermarked file without C2PA signature:")
                        with open(watermarked_path, "rb") as f:
                            st.download_button(
                                label="💾 Download Watermarked Audio",
                                data=f.read(),
                                file_name=f"{output_filename}.watermarked.wav",
                                mime="audio/wav"
                            )
                            st.caption(f"File will be saved as: **{output_filename}.watermarked.wav** in your Downloads folder")
                    else:
                        st.success("✓ Step 3: C2PA signature applied")
                        
                        # Provide download
                        with open(signed_path, "rb") as f:
                            st.download_button(
                                label="💾 Download Authenticated Audio",
                                data=f.read(),
                                file_name=f"{output_filename}.signed.wav",
                                mime="audio/wav"
                            )
                            st.caption(f"File will be saved as: **{output_filename}.signed.wav** in your Downloads folder")
                    
                    # Step 4: Store manifest in local database
                    if add_manifest_to_store(manifest_id, metadata):
                        st.success(f"✓ Step 4: Manifest ID {manifest_id} stored in database")
                    else:
                        st.warning("⚠️ Could not store manifest in database")
                    
                    # Clean up temp files
                    for temp_path in [watermarked_path, manifest_path]:
                        if temp_path.exists():
                            temp_path.unlink()
                    
                    # Clear processing state
                    del st.session_state.process_audio
                            
                except Exception as e:
                    st.error(f"Error during authentication process: {e}")
                    # Clear processing state on error
                    if 'process_audio' in st.session_state:
                        del st.session_state.process_audio

else:
    # TAMPERING MODE
    st.markdown("---")
    st.info("📝 **Purpose:** This tool adds controlled tampering to audio files so you can test the verification system's ability to detect modifications.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📎 Upload Audio to Tamper")
        
        tamper_uploaded_file = st.file_uploader(
            "Choose an audio file to add tampering:",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Upload any audio file to add 10 seconds of noise at the end",
            key="tamper_uploader"
        )
        
        if tamper_uploaded_file is not None:
            # Handle tamper temp file
            file_bytes = tamper_uploaded_file.getvalue()
            file_key = f"{tamper_uploaded_file.name}_{len(file_bytes)}"
            
            if 'tamper_current_file_key' not in st.session_state or st.session_state.tamper_current_file_key != file_key:
                # Clean up old temp file if it exists
                if 'tamper_temp_file_path' in st.session_state and st.session_state.tamper_temp_file_path and Path(st.session_state.tamper_temp_file_path).exists():
                    try:
                        os.unlink(st.session_state.tamper_temp_file_path)
                    except:
                        pass
                
                # Create new temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(tamper_uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(file_bytes)
                    st.session_state.tamper_temp_file_path = tmp_file.name
                    st.session_state.tamper_current_file_key = file_key
            
            tamper_file_path = Path(st.session_state.tamper_temp_file_path)
            
            st.success(f"✓ File loaded: {tamper_uploaded_file.name}")
            
            # Check audio info
            try:
                with sf.SoundFile(tamper_file_path, "r") as f:
                    duration = len(f) / f.samplerate
                    st.info(f"Original Duration: {duration:.1f} seconds, Sample rate: {f.samplerate} Hz")
                    st.info(f"After tampering: {duration + 10:.1f} seconds (+10s noise)")
            except:
                st.error("Error reading audio file.")
                tamper_file_path = None
            
            st.subheader("🔊 Audio Preview")
            st.audio(tamper_uploaded_file)
        else:
            tamper_file_path = None
            st.info("Please upload an audio file to add tampering")
    
    with col2:
        st.subheader("⚙️ Tampering Controls")
        
        # Output filename
        tamper_output_filename = st.text_input(
            "Output Filename*", 
            placeholder="e.g., court1", 
            help="Base name for the tampered file (without extension)",
            key="tamper_filename"
        )
        
        if tamper_output_filename:
            st.info(f"💾 Output will be: **{tamper_output_filename}.tampered.wav**")
        
        # Tampering settings
        st.markdown("**Tampering Settings**")
        noise_duration = st.slider("Noise Duration (seconds)", min_value=1, max_value=30, value=10, help="How many seconds of noise to add at the end")
        noise_level = st.slider("Noise Level", min_value=0.05, max_value=0.5, value=0.1, step=0.05, help="Intensity of the noise (0.05 = quiet, 0.5 = loud)")
        
        st.markdown("**What This Does:**")
        st.markdown(f"• Adds **{noise_duration} seconds** of random noise to the end")
        st.markdown("• Preserves original audio quality")
        st.markdown("• Creates detectable tampering for verification testing")
        
        # Add tampering button
        if st.button("⚙️ Add Tampering", type="primary", use_container_width=True):
            if not tamper_output_filename.strip():
                st.error("⚠️ Please enter an output filename")
            elif tamper_file_path is None:
                st.error("⚠️ Please upload an audio file first")
            else:
                # Store tampering data in session state
                st.session_state.process_tampering = {
                    'tamper_file_path': tamper_file_path,
                    'output_filename': tamper_output_filename,
                    'noise_duration': noise_duration,
                    'noise_level': noise_level
                }
                st.success("✓ Tampering queued - see results below!")
                st.rerun()
    
    # Process tampering outside of button context
    if 'process_tampering' in st.session_state and st.session_state.process_tampering:
        process_data = st.session_state.process_tampering
        tamper_file_path = process_data['tamper_file_path']
        output_filename = process_data['output_filename']
        noise_duration = process_data['noise_duration']
        noise_level = process_data['noise_level']
        
        st.markdown("---")
        st.subheader("⚙️ Adding Tampering...")
        
        # Option to cancel
        col_process, col_cancel = st.columns([3, 1])
        with col_cancel:
            if st.button("Cancel Tampering", type="secondary"):
                del st.session_state.process_tampering
                st.rerun()
        
        with st.spinner(f"Adding {noise_duration} seconds of noise tampering..."):
            try:
                # Create tampered file
                tampered_path = tamper_file_path.with_suffix('.tampered.wav')
                
                # Use updated function with noise level
                success, message = add_tampering_to_audio(str(tamper_file_path), str(tampered_path), noise_duration, noise_level)
                
                if not success:
                    st.error(f"Tampering failed: {message}")
                    del st.session_state.process_tampering
                    st.stop()
                
                st.success(f"✓ {message}")
                
                # Provide download
                with open(tampered_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Tampered Audio",
                        data=f.read(),
                        file_name=f"{output_filename}.tampered.wav",
                        mime="audio/wav"
                    )
                    st.caption(f"File will be saved as: **{output_filename}.tampered.wav** in your Downloads folder")
                
                st.info("🗺 **Next Steps:** Switch to 'Verify Audio' mode and upload this tampered file to test the detection system!")
                
                # Clean up temp files
                if tampered_path.exists():
                    tampered_path.unlink()
                
                # Clear processing state
                del st.session_state.process_tampering
                        
            except Exception as e:
                st.error(f"Error during tampering process: {e}")
                if 'process_tampering' in st.session_state:
                    del st.session_state.process_tampering

st.divider()

with st.expander("📖 How to Use This System"):
    st.markdown("""
    ### Getting Started
    
    **Step 1: Upload Your Court Audio File**
    - Click "Browse files" or drag and drop your court recording
    - Supported formats: WAV, MP3, M4A, FLAC, OGG
    - You'll see the audio player once uploaded
    
    **Step 2: Choose Your Verification Method**
    
    You have three verification options:
    
    1. **📜 View Content Credentials**
       - Click this to check the digital seal
       - Shows who created the file and when
       - Best for: Verifying official court recordings
    
    2. **🔐 Check Watermark**
       - Click this to scan for hidden watermarks
       - Detects any tampering in the audio
       - Best for: Finding edited sections
    
    3. **🔗 Check Durable Credentials**
       - Click this for our most advanced check
       - Recovers information even from damaged files
       - Best for: Court files from social media or unknown sources
    
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
    ## Welcome to the Court Audio Authentication System
    
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
