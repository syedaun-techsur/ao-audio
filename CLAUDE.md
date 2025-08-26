# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research-phase MVP** exploring solutions for judicial media authentication to combat AI-generated deepfakes. The project addresses critical concerns from judges about AI-altered audio/video content using their voices and the need for tamper-proof authentication systems.

### Problem Statement

Judges face a two-fold authentication problem:
1. **Prevention**: Protecting authentic court recordings from voice extraction and AI manipulation
2. **Detection**: Identifying AI-generated content that impersonates judicial voices or inserts fake segments

Examples of threats:
- Extracting judge voices to create malicious deepfake videos/calls  
- Inserting fake 10-second segments into real judicial opinions
- AI-generated content that appears authentic but damages judicial integrity

### Research Approach

The system explores **Durable Content Credentials** - a novel three-layer approach:
1. **AudioSeal Watermarking**: Embeds imperceptible watermarks with manifest IDs
2. **C2PA Content Credentials**: Cryptographic signatures with court metadata  
3. **Durable Recovery**: Watermark-embedded authentication that survives metadata stripping

This is experimental - testing different tools and techniques to find optimal solutions.

## Architecture

### Core Components

- `streamlit_app.py`: Web interface for testing verification approaches
- `embed_audioseal_with_id.py`: Embeds 16-bit manifest IDs in watermarks (0-65535)
- `detect_audioseal_with_id.py`: Recovers credentials even from tampered audio
- `create_tampered_version.py`: Simulates AI insertion attacks
- `manifest.json`: Court-specific C2PA metadata template

### Research Test Files

- `hearing.raw.wav`: Original court audio (47:32) - baseline
- `hearing.signed.wav`: Watermarked + C2PA signed - authenticated version
- `hearing.tampered.wav`: AI-insertion simulation (10s fake content at end)

## Development Commands

### Web Interface for Testing
```bash
# Run experimental verification system
streamlit run streamlit_app.py
# Test at http://localhost:8501
```

### AudioSeal Watermarking Experiments
```bash
# Test watermark embedding with court manifest ID
python embed_audioseal_with_id.py --in hearing.raw.wav --out hearing.wm.wav --manifest-id 12345

# Test credential recovery from tampered audio
python detect_audioseal_with_id.py --in hearing.tampered.wav

# Simulate AI insertion attack
python create_tampered_version.py
```

### C2PA Content Credentials Testing
```bash
# Verify judicial signatures and metadata
c2patool hearing.signed.wav --info

# Sign court audio with judicial credentials
c2patool hearing.wm.wav -m manifest.json -o hearing.signed.wav

# Deep inspection of court metadata
c2patool hearing.signed.wav --detailed
```

## Dependencies

### System Requirements
- Python 3.8+ with pip
- FFmpeg for audio processing
- c2patool for content credentials verification

### Research Dependencies
```bash
pip install torch soundfile numpy audioseal streamlit
```

**Note**: SafeSpeech/ contains extensive audio synthesis models for deepfake research (see SafeSpeech/requirements.txt).

## Technical Constraints

### Audio Processing Requirements
- 16kHz mono WAV format for watermarking experiments
- 16-bit manifest ID capacity (65,536 unique court cases)
- Chunk processing: 10-30 second segments for large files
- CPU-intensive: uses half available threads for processing

### Research Limitations
- **Prototype status**: Mock database for court manifests
- **Development certificates**: Not production judicial signatures
- **Limited testing**: Small sample of court audio types
- **Processing time**: Large files take several minutes

## Mock Court Database Structure

Research uses simulated judicial manifest database:

```python
manifests_db = {
    12345: {
        "creator": "Official Court Recording",
        "court_id": "USDC-2024-001",
        "title": "Hearing on Case No. 2024-CV-00123", 
        "recorder": "Court Reporter Jane Smith",
        "location": "US District Court, Southern District",
        "date": "2024-01-15"
    }
}
```

## Research Security Features

- **Metadata stripping resilience**: Watermarks survive social media upload
- **Tamper localization**: Identifies exact timestamps of AI insertions
- **Dual verification**: Both C2PA and watermark must validate
- **Deepfake detection**: Distinguishes authentic vs. AI-generated segments
- **Voice extraction protection**: Makes unauthorized voice cloning detectable

## Research Phase Notes

- Exploring multiple authentication approaches simultaneously
- Testing resilience against various AI manipulation techniques  
- Evaluating different watermarking vs. signature-based solutions
- Investigating blockchain integration for court manifest storage
- Prototype validates concepts - not ready for judicial production use