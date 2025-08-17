Implementation flow

Prepare the signed master locally
• Install c2patool
• Run the watermark embedder on the raw file → output: hearing.wm.wav (or .m4a).
• Create manifest.json (title, court id, date/time).
• Embed: c2patool hearing.wm.wav -m manifest.json -o hearing.signed.m4a
• Verify once: c2patool hearing.signed.m4a --info 

Stand up the demo web app
• Build a Streamlit app (run locally).
• Implement UI in streamlit_app.py:
– Play hearing.signed.wav via st.audio using its local file path (and hearing.tampered.wav likewise).
– "View Content Credentials" panel: run c2patool --info on the local file and display output.
– "Check Watermark" button: runs detect_audioseal on the local file and displays status, score, and tampered regions.

Implement watermark verification (Streamlit)
• Server-side function in the Streamlit app:
– Input: local file path 
– Run AudioSeal verifier on the local file, detect gaps in watermark.
– Check for suspicious gaps (>= 1.0s duration) rather than just overall score.
– Return: { 
    status: "AUTHENTIC" | "TAMPERED", 
    score: <number>, 
    tampered_regions: [{"start_time": <s>, "end_time": <s>, "duration": <s>}],
    total_tampered_duration: <number>
  }.

Demonstration
Step 1: Play "Official Audio" in the Streamlit app.
Step 2: Click "View Content Credentials" → display signer, time, and manifest details via local c2patool --info (C2PA).
Step 3: Click "Check Watermark" → status AUTHENTIC; score above threshold with no tampered regions detected.
Step 4: Select/Open "Tampered Version" (same audio with a 10-second synthetic insertion).
Step 5: Click "View Content Credentials" → verification fails (no C2PA manifests found).
Step 6: Click "Check Watermark" → status TAMPERED; shows tampered regions (e.g., 2852–2862 seconds) and total tampered duration.
Step 7: Conclude: authentic releases exhibit both valid Content Credentials (C2PA) and no watermark gaps; manipulated copies fail C2PA verification and/or show tampered regions in watermark analysis.

Success criteria (POC acceptance)
• Official file plays in the app and shows valid Content Credentials (from c2patool --info).
• Check Watermark on the official file returns AUTHENTIC with score ≥ configured threshold (e.g., 95%) and no tampered regions.
• Tampered file loads but Content Credentials verification fails (no C2PA manifests).
• Check Watermark on the tampered file returns TAMPERED with tampered regions identified (e.g., a 2852–2862s gap).
• CLI parity: c2patool --info and detect_audioseal.py run locally match the UI results.

## Durable Content Credentials (POC Enhancement)

### Concept
Demonstrates how watermarks can recover C2PA credentials even when metadata is stripped. Since watermarks survive in untampered portions of audio, they can be used to retrieve the original manifest.

### Implementation for Demo

Add a third button: **"🔗 Check Durable Credentials"**

This button simulates manifest recovery by:
1. Checking if watermark is present (even partially)
2. If watermark detected, "recovering" the original C2PA manifest
3. Showing both the recovery status and any tampered regions

### Expected Outputs

**For Official Recording (hearing.signed.wav):**
```
✅ DURABLE CREDENTIALS INTACT

Watermark Status: ✅ Fully intact
C2PA Recovery: ✅ Original manifest verified

Creator: Official Court Recording
Date: 2024-01-15
Court ID: USDC-2024-001

Result: Full chain of custody maintained
```

**For Tampered Version (hearing.tampered.wav):**
```
⚠️ DURABLE CREDENTIALS PARTIALLY RECOVERED

Watermark Status: ⚠️ Partial (90% intact)
C2PA Recovery: ✅ Manifest recovered from watermark

Recovered Manifest:
Creator: Official Court Recording  
Date: 2024-01-15
Court ID: USDC-2024-001

⚠️ WARNING: Tampering detected
Tampered Region: 47:42 - 47:52 (10 seconds)

Result: Original source verified but content has been modified
```

### Why This Matters for the Demo

This shows that even when:
- C2PA metadata is stripped (social media upload)
- Audio is partially tampered (10s insertion)

The system can still:
- Recover the original court credentials from the watermark
- Identify exactly where tampering occurred
- Prove the original source while detecting modifications

This demonstrates the resilience of the dual-layer approach - watermarks enable credential recovery even after metadata stripping, while still detecting tampering.
