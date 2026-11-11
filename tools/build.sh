#!/usr/bin/env bash
# Full pipeline: music -> parallel frame render (with motion blur) -> master MP4
set -euo pipefail
cd "$(dirname "$0")/.."
FPS=30; SUB=2; DUR=33; JOBS=${JOBS:-3}
mkdir -p build
python3 tools/music.py
cp build/music.wav ad/music.wav
# split the timeline into JOBS segments rendered in parallel
pids=()
for ((j=0; j<JOBS; j++)); do
  t0=$(python3 -c "print(round($DUR*$j/$JOBS*$FPS)/$FPS)"); t1=$(python3 -c "print(round($DUR*($j+1)/$JOBS*$FPS)/$FPS)")
  SUB=$SUB node tools/render.cjs video $FPS $t0 $t1 build/seg_$j.mp4 > build/seg_$j.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
: > build/segs.txt
for ((j=0; j<JOBS; j++)); do echo "file 'seg_$j.mp4'" >> build/segs.txt; done
# blend sub-frame pairs -> 30 fps with motion blur, add grade-safe sharpen, mux loudness-normalised audio
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segs.txt -i build/music.wav \
  -filter_complex "[0:v]tmix=frames=2:weights='1 1',select='mod(n\,2)',setpts=N/($FPS*TB),fps=$FPS,format=yuv420p[v];[1:a]loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset slow -crf 17 -profile:v high -level 4.2 -movflags +faststart \
  -c:a aac -b:a 320k -t $DUR out/Casio_ArabicDial_Spot_33s_1080p.mp4
echo done
