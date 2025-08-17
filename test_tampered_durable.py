#!/usr/bin/env python3
"""Test durable credentials on tampered file"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import check_durable_credentials

def test_tampered():
    file_path = "phone_recording_id_12345_tampered.wav"
    
    print("=" * 60)
    print("TESTING DURABLE CREDENTIALS ON TAMPERED FILE")
    print("=" * 60)
    print(f"\n📁 File: {file_path}")
    print("-" * 40)
    
    result = check_durable_credentials(file_path)
    
    print(f"Status: {result['status']}")
    
    if result["status"] == "RECOVERED":
        print(f"\n✅ Manifest Successfully Recovered!")
        print(f"   Manifest ID: {result.get('manifest_id', 'N/A')}")
        print(f"   Watermark Integrity: {result.get('watermark_integrity', 'N/A')}%")
        
        if result.get("manifest"):
            m = result["manifest"]
            print(f"\n📋 Recovered Manifest:")
            print(f"   Creator: {m['creator']}")
            print(f"   Date: {m['date']}")
            print(f"   Court ID: {m['court_id']}")
            print(f"   Title: {m['title']}")
            print(f"   Recorder: {m['recorder']}")
            print(f"   Location: {m['location']}")
        
        if result.get("tampered_regions"):
            print(f"\n⚠️ TAMPERING DETECTED:")
            print(f"   Number of tampered regions: {len(result['tampered_regions'])}")
            print(f"   Total tampered duration: {result.get('total_tampered_duration', 0)}s")
            
            for i, region in enumerate(result["tampered_regions"], 1):
                start = region['start_time']
                end = region['end_time']
                duration = region['duration']
                
                # Convert to minutes:seconds format
                start_min = int(start // 60)
                start_sec = start % 60
                end_min = int(end // 60)
                end_sec = end % 60
                
                print(f"   Region {i}: {start_min:02d}:{start_sec:05.2f} - {end_min:02d}:{end_sec:05.2f} ({duration}s)")
            
            print(f"\n📊 Summary:")
            print(f"   ✅ Original source verified (manifest recovered)")
            print(f"   ⚠️ Content has been modified")
            print(f"   → This demonstrates that even with tampering, we can:")
            print(f"      1. Recover the original credentials")
            print(f"      2. Identify exactly where tampering occurred")
        else:
            print(f"\n✅ No tampering detected - full integrity maintained")
    else:
        print(f"❌ Failed to recover manifest: {result.get('status')}")
        if result.get('error'):
            print(f"   Error: {result['error']}")

if __name__ == "__main__":
    test_tampered()