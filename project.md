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
