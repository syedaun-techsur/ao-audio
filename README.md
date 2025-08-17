# Media Provenance MVP with Durable Content Credentials

This project demonstrates a breakthrough in media authentication using three layers of verification for court audio recordings. The system introduces **Durable Content Credentials** - a novel approach that recovers authentication credentials even when traditional metadata is stripped.

## 🎯 Project Overview

The system combines three complementary technologies:
- **AudioSeal Watermarking**: Imperceptible watermarks that detect tampering at sample level
- **C2PA Content Credentials**: Cryptographic signatures proving origin and authenticity  
- **Durable Content Credentials**: Watermark-embedded manifest IDs that enable credential recovery

### 🚀 Key Innovation: Durable Content Credentials

Traditional authentication fails when files are uploaded to social media or processed by platforms that strip metadata. Our system embeds a 16-bit manifest ID (0-65535) within the AudioSeal watermark that links to the full C2PA manifest, enabling:

- **Credential recovery** even when C2PA metadata is stripped
- **Tampering detection** while preserving original source verification
- **Resilient authentication** that survives platform processing

## 🏗️ System Architecture

```
┌─────────────────┐
│  Raw Audio File │
│ (hearing.raw)   │  
└────────┬────────┘
         │
         v
┌─────────────────────────────────┐
│  1. AudioSeal Watermarking       │
│  - Embeds manifest ID (12345)    │
│  - Imperceptible modification    │
│  - Links to credential database  │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  2. C2PA Signing                 │
│  - Adds cryptographic signature  │
│  - Embeds court metadata         │
│  - Creates signed file           │
└────────┬────────────────────────┘
         │
         v
┌─────────────────┐
│  Signed Audio   │
│(hearing.signed) │
└─────────────────┘
```

### Three-Layer Verification System

1. **🔐 Check Watermark**: Detects AudioSeal watermarks and tampering
2. **📜 View Content Credentials**: Validates C2PA signatures and metadata  
3. **🔗 Check Durable Credentials**: Recovers credentials from watermark even when C2PA is stripped

## 🔧 Installation & Setup

### Prerequisites

```bash
# Python 3.8+ with pip
python --version

# FFmpeg for audio processing
brew install ffmpeg  # macOS
# or
apt-get install ffmpeg  # Linux

# C2PA tool for content credentials
brew install c2pa/c2patool/c2patool  # macOS
# or download from https://github.com/contentauth/c2patool
```

### Python Dependencies

```bash
pip install torch soundfile numpy audioseal streamlit
```

### Running the Application

```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py

# Access at: http://localhost:8501
```

## 📁 File Structure

The system uses three main audio files:

| File | Size | Duration | Description |
|------|------|----------|-------------|
| `hearing.raw.wav` | 87M | 47:32 | Original unprocessed court audio |
| `hearing.signed.wav` | 87M | 47:32 | Watermarked + C2PA signed version |
| `hearing.tampered.wav` | 87M | 47:42 | Modified with 10s audible noise at end |

### Core Scripts

```
streamlit_app.py              # Web interface with all verification tools
embed_audioseal_with_id.py    # Watermark embedding with manifest ID
detect_audioseal_with_id.py   # Watermark detection & ID extraction  
create_tampered_version.py    # Generate tampered test files
manifest.json                 # C2PA manifest template
```

## 📊 Verification Results

### Official Recording (hearing.signed.wav)
```
✅ C2PA: Valid signatures found
✅ Watermark: 100% intact, Manifest ID 12345
✅ Durable Credentials: Fully recoverable
```

### Tampered Recording (hearing.tampered.wav)
```
❌ C2PA: No manifests (stripped by modification)
⚠️ Watermark: Tampering detected at 47:32-47:42
✅ Durable Credentials: Manifest ID 12345 still recoverable
```

**Key Insight**: Even when C2PA metadata is stripped and audio is partially tampered, the system can still recover original credentials and identify exactly where tampering occurred.

## 🎵 AudioSeal Watermarking with Manifest IDs

### Embedding Watermark with Manifest ID
```bash
# Embed watermark with specific manifest ID
python embed_audioseal_with_id.py \
    --in hearing.raw.wav \
    --out hearing_watermarked.wav \
    --manifest-id 12345

# Parameters:
# --manifest-id: 16-bit ID (0-65535) linking to credential database
```

### Detecting Watermark and Extracting Manifest ID
```bash
# Extract manifest ID and recover credentials
python detect_audioseal_with_id.py --in hearing.signed.wav

# Output includes:
# - Detection probability
# - Extracted manifest ID
# - Retrieved C2PA manifest data
```

### Traditional Watermark Detection (Tampering Only)
```bash
# Detect tampering without credential recovery
python detect_audioseal.py --in hearing.tampered.wav --min-gap-sec 1.0

# Shows tampered regions with timestamps
```

## 🔐 C2PA Content Credentials Commands

### C2PA Verification
```bash
# Verify C2PA signatures and metadata
c2patool hearing.signed.wav --info

# Detailed C2PA manifest information
c2patool hearing.signed.wav --detailed

# Verify tampered file (should fail)
c2patool hearing.tampered.wav --info
```

### C2PA Manifest Creation
```bash
# Create signed audio with C2PA manifest
c2patool hearing.wm.wav -m manifest.json -o hearing.signed.wav
```

## 🔍 Three-Layer Verification Comparison

| Aspect | AudioSeal Watermarking | C2PA Content Credentials | Durable Content Credentials |
|--------|----------------------|---------------------------|------------------------------|
| **Purpose** | Tamper detection | Digital provenance | Credential recovery |
| **Resilience** | Survives compression | Fails when metadata stripped | Survives metadata stripping |
| **Tampering** | Localizes tampered regions | Binary pass/fail | Shows tampering + recovers source |
| **Output** | Confidence scores, gaps | Valid/Invalid, metadata | Recovered credentials + tampering |
| **Use Case** | "Is this audio authentic?" | "Who created this?" | "What was the original source?" |

## 🔗 Durable Content Credentials - Technical Details

### How It Works

1. **Embedding Phase**: 16-bit manifest ID embedded in AudioSeal watermark
2. **Storage**: ID maps to full C2PA manifest in database/blockchain
3. **Recovery**: Extract ID from watermark → lookup manifest → display credentials

### Manifest Database (Mock Implementation)

```python
manifests_db = {
    12345: {
        "creator": "Official Court Recording",
        "date": "2024-01-15", 
        "court_id": "USDC-2024-001",
        "title": "Hearing on Case No. 2024-CV-00123",
        "recorder": "Court Reporter Jane Smith",
        "location": "US District Court, Southern District"
    }
}
```

### Why This Matters

- **Social Media Resilience**: Credentials survive platform processing
- **Tamper Transparency**: Shows both original source AND modifications
- **Future-Proof**: Works even when new tampering methods defeat C2PA

## 📊 Verification Results

## 💻 Web Interface Features

### Upload Section
- **Drag & Drop**: Support for WAV, MP3, M4A, FLAC, OGG formats
- **Audio Player**: Preview uploaded files before verification
- **File Information**: Shows file size and duration

### Three Verification Buttons

#### 📜 View Content Credentials
- Checks C2PA signatures using c2patool
- Displays creator, date, court ID, case information
- Shows validation status (Valid/Invalid/Not Found)

#### 🔐 Check Watermark  
- Detects AudioSeal watermarks
- Identifies tampered regions with precise timestamps
- Shows integrity percentage and gap duration

#### 🔗 Check Durable Credentials (NEW)
- **For Intact Files**: Shows full credential recovery
- **For Tampered Files**: 
  - Prominently displays tampering warning
  - Lists all tampered regions with timestamps
  - Shows recovered original credentials below
  - Demonstrates resilience of the system

### Sample Output for Tampered File
```
⚠️ TAMPERING DETECTED - CREDENTIALS RECOVERED

⚠️ Tampered Regions Found:
Region 1: 47:32 - 47:34 (2.42 seconds)
Region 2: 47:34 - 47:38 (4.18 seconds) 
Region 3: 47:40 - 47:42 (2.81 seconds)

Total Tampered Duration: 9.41 seconds

✅ Original C2PA Credentials Successfully Recovered:
- Creator: Official Court Recording
- Date: 2024-01-15
- Court ID: USDC-2024-001
- Title: Hearing on Case No. 2024-CV-00123
- Recorder: Court Reporter Jane Smith
- Location: US District Court, Southern District
```

## 🔧 Troubleshooting

### c2patool Issues
```bash
# Check if c2patool is in PATH
which c2patool

# Remove macOS quarantine if needed
xattr -d com.apple.quarantine ~/bin/c2patool

# Test c2patool
c2patool -V
```

### Watermarking Issues
```bash
# Check Python dependencies
pip list | grep audioseal

# Verify audio file format
file hearing.raw.wav
```

## 🎯 Use Cases

1. **Court Proceedings**: Verify audio evidence hasn't been altered
2. **News Media**: Authenticate leaked recordings after social media upload
3. **Legal Discovery**: Prove chain of custody for audio evidence
4. **Corporate Security**: Verify authenticity of recorded meetings
5. **Content Moderation**: Detect deepfake or manipulated audio

## 🔒 Security Features

- **Resilient to Metadata Stripping**: Watermarks survive social media processing
- **Tamper Localization**: Identifies exact timestamps of modifications  
- **Dual Verification**: Both C2PA and watermark must validate for full trust
- **Cryptographic Integrity**: Uses industry-standard signing certificates
- **Future-Proof Design**: Recovers credentials even when C2PA fails

## 🚧 Current Limitations

1. **Prototype Status**: Mock database for manifest storage
2. **Message Capacity**: Limited to 16-bit IDs (65,536 unique manifests)
3. **Audio Format**: Requires 16kHz mono for optimal watermarking
4. **Processing Time**: Large files take several minutes to process
5. **Development Certificates**: Uses test signing keys (not production-ready)

