# Media Provenance MVP - Court Audio Authenticity POC

This project demonstrates end-to-end authenticity for official court audio files using both **AudioSeal watermarking** and **C2PA Content Credentials**. The POC shows that published files carry verifiable provenance and that tampered copies fail verification.

## 🎯 Purpose

Demonstrate a complete authenticity verification system for court audio files with:
- **AudioSeal Watermarking**: Invisible audio watermarking for tamper detection
- **C2PA Content Credentials**: Digital signatures and metadata for provenance
- **Web-based verification**: Browser-based verification interface
- **Azure hosting**: Cloud storage and web application deployment

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Audio     │───▶│  Watermarked    │───▶│  C2PA Signed    │
│  (hearing.raw)  │    │  (hearing.wm)   │    │  (hearing.signed)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Watermark Embed │    │ Watermark Check │    │ C2PA Verify     │
│ embed_audioseal │    │ detect_audioseal│    │ c2patool --info │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Prerequisites

### Required Tools
- **c2patool**: C2PA command-line tool for Content Credentials
- **Python 3.8+**: For AudioSeal watermarking
- **ffmpeg**: For audio processing and tampered file creation
- **openssl**: For certificate generation (if needed)

### Python Dependencies
```bash
pip install torch soundfile numpy audioseal
```

## 🚀 Step 1: Local Setup (COMPLETED ✅)

### What We Accomplished

**Step 1** successfully created a complete local signing pipeline with:

1. **✅ c2patool Installation**
   - Downloaded and installed c2patool v0.19.1
   - Added to PATH: `~/bin/c2patool`
   - Removed macOS quarantine restrictions

2. **✅ C2PA Manifest Creation**
   - Created `manifest.json` with court metadata:
     - Court identifier: Superior Court of California, San Francisco
     - Case number: CV-2024-001234
     - Judge: Hon. Jane Smith
     - Recording equipment and location details
     - Timestamp and digital source information

3. **✅ Audio Signing Process**
   - Input: `hearing.wm.wav` (watermarked audio)
   - Output: `hearing.signed.wav` (C2PA signed)
   - Used development signing certificate
   - Manifest size: 14,266 bytes (0.02% of file)

4. **✅ Tampered File Creation**
   - Created `hearing.tampered.wav` with 10-second synthetic audio append
   - File size: 91,609,934 bytes (vs 91,304,174 for signed)
   - **No C2PA manifests** - perfect for testing verification failure

5. **✅ Verification Testing**
   - Official file: ✅ Valid C2PA manifest, properly signed
   - Tampered file: ❌ No C2PA manifests (verification fails)

### Files Created in Step 1

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `hearing.signed.wav` | 91,304,174 bytes | Official C2PA signed audio | ✅ Valid |
| `hearing.tampered.wav` | 91,609,934 bytes | Tampered version for testing | ❌ Invalid |
| `manifest.json` | 1,389 bytes | C2PA manifest with court metadata | ✅ Created |
| `verification-output.json` | 116 bytes | C2PA verification results | ✅ Generated |

### Running Step 1

```bash
# Make script executable
chmod +x setup-step1.sh

# Run Step 1 (already completed)
./setup-step1.sh
```

## 🎵 AudioSeal Watermarking Commands

### Watermark Embedding
```bash
# Embed watermark into raw audio
python embed_audioseal.py
# Input: hearing.raw.wav
# Output: hearing.wm.wav
```

### Watermark Detection
```bash
# Detect watermark in watermarked file
python detect_audioseal.py --in hearing.wm.wav

# Detect watermark in tampered file (shows gaps)
python detect_audioseal.py --in attacked.wav --chunk-sec 5 --min-gap-sec 1.0
```

### Watermark Detection Parameters
- `--chunk-sec`: Analysis window size (default: 10s)
- `--min-gap-sec`: Minimum gap to report (default: 2.0s)
- `--in`: Input file path

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

## 🔍 Key Differences: Watermarking vs C2PA

| Aspect | AudioSeal Watermarking | C2PA Content Credentials |
|--------|----------------------|-------------------------|
| **Purpose** | Tamper detection | Digital provenance & signatures |
| **Detection** | Analyzes audio content for watermark presence | Validates cryptographic signatures |
| **Output** | Confidence scores, gap detection | Valid/Invalid, metadata display |
| **Tamper Response** | Shows where watermark is missing | Shows signature verification failure |
| **File Format** | Works with any audio format | Embeds manifest in supported formats |
| **Use Case** | "Is this audio authentic?" | "Who created this and when?" |

### Combined Verification Strategy

**For complete authenticity verification:**
1. **C2PA Check**: Verify digital signature and provenance
2. **Watermark Check**: Verify audio content integrity
3. **Both must pass** for full authenticity validation

## 📊 Verification Results

### Official File (`hearing.signed.wav`)
```
C2PA: ✅ Valid manifest, properly signed
Watermark: ✅ High confidence watermark detection
Status: AUTHENTIC
```

### Tampered File (`hearing.tampered.wav`)
```
C2PA: ❌ No C2PA manifests found
Watermark: ⚠️ Gap detection in tampered regions
Status: TAMPERED
```

## 🚀 Next Steps

### Step 2: Azure Storage Setup
- Create Azure Storage account with `court-release` container
- Upload signed and tampered audio files
- Generate SAS URLs for web access

### Step 3: Web Application
- Create Azure Static Web App
- Build HTML interface with audio player
- Implement C2PA verification in browser
- Add watermark verification endpoint

### Step 4: Demo Interface
- Audio playback with Content Credentials display
- Real-time verification status
- Side-by-side comparison of authentic vs tampered files

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

## 📝 Notes

- **Development Certificates**: Using default development signing for POC
- **Production**: Requires proper certificate authority for production use
- **File Formats**: C2PA supports M4A, MP4, WAV, and other formats
- **Watermarking**: AudioSeal works with 16kHz mono WAV files

---

**Status**: Step 1 Complete ✅ | Step 2: Azure Setup (Next)
