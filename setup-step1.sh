#!/bin/bash

echo "=== Step 1: Prepare the signed master locally ==="

# Check if c2patool is installed
if ! command -v c2patool &> /dev/null; then
    echo "❌ c2patool not found. Please install it first:"
    echo "   Option 1: cargo install c2patool"
    echo "   Option 2: Download from https://github.com/contentauth/c2pa-rs/releases"
    exit 1
fi

echo "✅ c2patool found: $(c2patool -V)"

echo "ℹ️  Using default development signing (no certificate files needed)"

# Check if watermarked file exists
if [ ! -f "hearing.wm.wav" ]; then
    echo "❌ hearing.wm.wav not found. Please run your watermarking process first."
    exit 1
fi

echo "✅ Watermarked audio found: hearing.wm.wav"

# Create signed audio file
echo "🔐 Creating signed audio file..."
c2patool hearing.wm.wav -m manifest.json -o hearing.signed.wav

if [ $? -eq 0 ]; then
    echo "✅ Signed audio created: hearing.signed.wav"
else
    echo "❌ Failed to create signed audio file"
    exit 1
fi

# Verify the signed file
echo "🔍 Verifying signed audio..."
c2patool hearing.signed.wav --info > verification-output.json

if [ $? -eq 0 ]; then
    echo "✅ Verification complete. Output saved to: verification-output.json"
    echo "📋 Verification summary:"
    c2patool hearing.signed.wav --info | head -20
else
    echo "❌ Verification failed"
    exit 1
fi

# Create tampered version for testing
echo "🔨 Creating tampered version for testing..."

# Check if ffmpeg is available
if command -v ffmpeg &> /dev/null; then
    # Generate 10 seconds of synthetic audio (sine wave)
    ffmpeg -f lavfi -i "sine=frequency=1000:duration=10" -ar 16000 -ac 1 -y synthetic_tail.wav
    
    # Append synthetic audio to create tampered version
    ffmpeg -i hearing.signed.wav -i synthetic_tail.wav -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" -y hearing.tampered.wav
    
    # Cleanup
    rm synthetic_tail.wav
    
    echo "✅ Tampered audio created: hearing.tampered.wav"
else
    echo "⚠️  ffmpeg not found. Creating simple tampered version using existing file..."
    # Simple approach: copy and append some data
    cp hearing.signed.wav hearing.tampered.wav
    echo "Tampered data" >> hearing.tampered.wav
    echo "✅ Simple tampered audio created: hearing.tampered.wav"
fi

echo ""
echo "🎉 Step 1 Complete! Files created:"
echo "   📄 manifest.json - C2PA manifest with court metadata"
echo "   🎵 hearing.signed.wav - Official signed audio"
echo "   🎵 hearing.tampered.wav - Tampered version for testing"
echo "   📋 verification-output.json - Verification details"
echo ""
echo "Next: Run Step 2 to set up Azure storage and upload artifacts" 