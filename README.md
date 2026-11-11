# CASIO · Arabic Dial Collection — 33 s spec spot

Spec (non-commissioned) commercial for the Casio **MTP-E735AR** "Arabic Dial"
series: five colourways (black, royal blue, teal, ice blue, silver) with
Eastern Arabic numerals and an embossed arabesque dial.

**Deliverable:** [`out/Casio_ArabicDial_Spot_33s_1080p.mp4`](out/Casio_ArabicDial_Spot_33s_1080p.mp4)
1920×1080 · 30 fps · H.264 + AAC 320 kbps · −14 LUFS (web/social loudness).

## Creative direction — "Time speaks your language" · الوقت يتكلم لغتك

Upbeat Arabic-pop / electro at 120 BPM in maqam Hijaz. Every cut, wipe and
seconds-hand tick lands on the beat grid shared by the music and the picture.

| Time | Scene | Music |
|---|---|---|
| 0–4 s | Dark, an 8-point-star relief lit by a moving light. The numerals ١ → ١٢ are written round a dial, one per clock tick, and the ticks speed up from quarter notes to eighth notes. | Ticks, drone, oud run, riser |
| 4–6 s | Match cut to the real **black** watch, "ARABIC DIAL" title, light sweep | **Drop**: kick, darbuka maqsum, bass, qanun hook |
| 6–10 s | Colour run: **royal → teal → ice → silver**, each with its own wipe and camera move, plus big metallic numerals in parallax | Whooshes on each change |
| 10–12 s | Push into the silver dial: TIME / SPEAKS / YOUR / LANGUAGE (one word per beat) | |
| 12–20 s | 4 macro features: Eastern Arabic numerals · arabesque dial · **bilingual day wheel (MON → الاثنين)** · stainless steel / 50 m | Ney-style lead melody, build |
| 20–24 s | The key visual in motion: 5 colour panels slam in, one per eighth note | **Drop 2** + riq |
| 24–26 s | Macro cuts on the beat: numerals, crown, bracelet, hands | Stabs |
| 26–28 s | Arabic calligraphy written right to left: الوقت يتكلم لغتك | Breakdown, oud |
| 28–33 s | Pack shot of all 5 watches + CASIO end card | Final hit, shimmer |

The seconds hand printed in the photo was digitally removed from each watch.
A new hand is drawn in its place and **ticks live on the beat**.

## How it is made (all reproducible)

```
assets/source/keyvisual.jpg   supplied key visual (5 watches)
tools/second_hand.py          removes the printed seconds hand (mirror-symmetry fill + inpaint)
tools/upscale.py              EDSR ×4 super-resolution → ad/img/watch_*.png (3× working size)
tools/relief.py               procedural Hankin 8-point-star strapwork relief textures
tools/music.py                the soundtrack, synthesised from scratch (no samples, royalty-free)
ad/index.html                 the motion design: deterministic GSAP timeline, seek(t) renders frame t
tools/render.cjs              headless Chromium frame renderer (2 sub-frames/frame → motion blur)
tools/build.sh                full pipeline → out/*.mp4
```

Cut-outs were made with `rembg` (BiRefNet). Fonts: Archivo, Reem Kufi, Amiri,
Aref Ruqaa (Google Fonts, OFL). Animation: GSAP 3.12.

**Live preview:** serve the `ad/` folder (e.g. `python3 -m http.server -d ad`)
and open it in a browser. Click or press Space to play with sound, and use ← / → to scrub.

**Rebuild:** `pip install numpy scipy numba pillow opencv-contrib-python-headless`,
then `tools/build.sh` (Node + Playwright Chromium + ffmpeg required).

> This is a spec / portfolio piece. CASIO and the product imagery belong to
> Casio Computer Co., Ltd. Public or commercial use needs the brand's approval.
