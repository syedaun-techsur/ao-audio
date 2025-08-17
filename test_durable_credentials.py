#!/usr/bin/env python3
"""Test script for durable credentials functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import check_durable_credentials, decode_message_to_id, get_mock_manifest_from_id

def test_files():
    """Test durable credentials on various files"""
    
    test_files = [
        ("phone_recording_id_12345.wav", "File with manifest ID 12345"),
        ("phone_recording_id_54321.wav", "File with manifest ID 54321"),
        ("phone_recording.wav", "Original file without manifest ID"),
    ]
    
    print("=" * 60)
    print("DURABLE CREDENTIALS TEST RESULTS")
    print("=" * 60)
    
    for file_path, description in test_files:
        if not os.path.exists(file_path):
            print(f"\n❌ File not found: {file_path}")
            continue
            
        print(f"\n📁 Testing: {file_path}")
        print(f"   Description: {description}")
        print("-" * 40)
        
        result = check_durable_credentials(file_path)
        
        if result["status"] == "ERROR":
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
        
        elif result["status"] == "NO_WATERMARK":
            print("   ❌ No watermark detected")
        
        elif result["status"] == "WATERMARK_ONLY":
            print(f"   ⚠️ Watermark detected but no manifest ID")
            print(f"   Watermark Integrity: {result['watermark_integrity']}%")
        
        elif result["status"] == "RECOVERED":
            print(f"   ✅ Manifest Recovered!")
            print(f"   Manifest ID: {result['manifest_id']}")
            print(f"   Watermark Integrity: {result['watermark_integrity']}%")
            
            if result["manifest"]:
                m = result["manifest"]
                print(f"   Creator: {m['creator']}")
                print(f"   Court ID: {m['court_id']}")
                print(f"   Date: {m['date']}")
            
            if result["tampered_regions"]:
                print(f"   ⚠️ Tampering detected: {len(result['tampered_regions'])} region(s)")
                print(f"   Total tampered duration: {result['total_tampered_duration']}s")

if __name__ == "__main__":
    test_files()