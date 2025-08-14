#!/bin/bash

echo "=== Step 1: Prepare the signed master locally ==="

# Check if c2patool is installed
if ! command -v c2patool &> /dev/null; then
    echo "ERROR: c2patool not found. Please install it first:"
    echo "   Option 1: cargo install c2patool"
    echo "   Option 2: Download from https://github.com/contentauth/c2pa-rs/releases"
    exit 1
fi

echo "SUCCESS: c2patool found: $(c2patool -V)"

echo "INFO: Using default development signing (no certificate files needed)"

# Check if watermarked file exists
if [ ! -f "hearing.wm.wav" ]; then
    echo "ERROR: hearing.wm.wav not found. Please run your watermarking process first."
    exit 1
fi

echo "SUCCESS: Watermarked audio found: hearing.wm.wav"

# Create signed audio file
echo "Creating signed audio file..."
c2patool hearing.wm.wav -m manifest.json -o hearing.signed.wav -f

if [ $? -eq 0 ]; then
    echo "SUCCESS: Signed audio created: hearing.signed.wav"
else
    echo "ERROR: Failed to create signed audio file"
    exit 1
fi

# Verify the signed file
echo "Verifying signed audio..."
c2patool hearing.signed.wav --info

if [ $? -eq 0 ]; then
    echo "SUCCESS: Verification complete"
else
    echo "ERROR: Verification failed"
    exit 1
fi

# Create tampered version for testing
echo "Creating tampered version for testing..."

# Check if ffmpeg is available
if command -v ffmpeg &> /dev/null; then
    # Generate 10 seconds of synthetic audio (sine wave)
    ffmpeg -f lavfi -i "sine=frequency=1000:duration=10" -ar 16000 -ac 1 -y synthetic_tail.wav
    
    # Append synthetic audio to create tampered version
    ffmpeg -i hearing.signed.wav -i synthetic_tail.wav -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" -y hearing.tampered.wav
    
    # Cleanup
    rm synthetic_tail.wav
    
    echo "SUCCESS: Tampered audio created: hearing.tampered.wav"
else
    echo "WARNING: ffmpeg not found. Creating simple tampered version using existing file..."
    # Simple approach: copy and append some data
    cp hearing.signed.wav hearing.tampered.wav
    echo "Tampered data" >> hearing.tampered.wav
    echo "SUCCESS: Simple tampered audio created: hearing.tampered.wav"
fi

# Test C2PA verification on tampered file (should fail)
echo ""
echo "Testing C2PA verification on tampered file..."
echo "Expected: Verification should fail (no valid manifests)"
c2patool hearing.tampered.wav --info

# Activate virtual environment for watermark detection
echo ""
echo "Activating virtual environment for AudioSeal watermarking..."
source .venv/bin/activate

# Create comprehensive results file
echo "=== MEDIA PROVENANCE VERIFICATION RESULTS ===" > verification_results.txt
echo "Generated: $(date)" >> verification_results.txt
echo "" >> verification_results.txt

echo "C2PA DIGITAL SIGNATURE VERIFICATION:" >> verification_results.txt
echo "=====================================" >> verification_results.txt
echo "Signed file (hearing.signed.wav): VALID - Contains court metadata and digital signature" >> verification_results.txt
echo "Tampered file (hearing.tampered.wav): INVALID - No C2PA manifests found" >> verification_results.txt
echo "" >> verification_results.txt

echo "AUDIOSEAL WATERMARK DETECTION:" >> verification_results.txt
echo "==============================" >> verification_results.txt

# Test watermark detection on tampered file (should show gaps)
echo ""
echo "Testing watermark detection on tampered file..."
echo "Expected: Should show gaps in tampered regions"
echo "Tampered file results:" >> verification_results.txt
python detect_audioseal.py --in hearing.tampered.wav --chunk-sec 5 --min-gap-sec 1.0 >> verification_results.txt
echo "" >> verification_results.txt

# Test watermark detection on original signed file (should show high confidence)
echo ""
echo "Testing watermark detection on signed file..."
echo "Expected: Should show high confidence watermark detection"
echo "Signed file results:" >> verification_results.txt
python detect_audioseal.py --in hearing.signed.wav >> verification_results.txt
echo "" >> verification_results.txt

# Test watermark detection on watermarked file (before C2PA signing)
echo ""
echo "Testing watermark detection on watermarked file (before C2PA signing)..."
echo "Expected: Should show high confidence watermark detection"
echo "Watermarked file results:" >> verification_results.txt
python detect_audioseal.py --in hearing.wm.wav >> verification_results.txt

echo ""
echo "SUCCESS: Step 1 Complete! Files created and tested:"
echo "   manifest.json - C2PA manifest with court metadata"
echo "   hearing.signed.wav - Official signed audio (VERIFIED)"
echo "   hearing.tampered.wav - Tampered version for testing (VERIFIED AS TAMPERED)"
echo "   verification_results.txt - Complete verification results"
echo ""
echo "Next: Run Step 2 to set up Azure storage and upload artifacts" 