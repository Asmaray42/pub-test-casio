// Frame-accurate renderer: drives ad/index.html with window.seek(t) in headless
// Chromium and pipes JPEG frames to ffmpeg.
//   node tools/render.cjs stills 0.5,4.3,...     -> build/stills/t_XX.jpg
//   node tools/render.cjs video FPS START END OUT -> raw segment (no audio)
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path'), fs = require('fs');
const ROOT = path.resolve(__dirname, '..');
(async () => {
  const [mode, a, b, c, d] = process.argv.slice(2);
  const browser = await chromium.launch({ args: ['--allow-file-access-from-files', '--disable-web-security', '--force-color-profile=srgb', '--hide-scrollbars'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.error('[page]', m.text()); });
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  await page.goto('file://' + path.join(ROOT, 'ad/index.html') + '?render');
  await page.waitForFunction(() => window.READY === true, null, { timeout: 120000 });
  if (mode === 'stills') {
    const dir = path.join(ROOT, process.env.STILLS || 'build/stills'); fs.mkdirSync(dir, { recursive: true });
    for (const t of a.split(',').map(Number)) {
      await page.evaluate(t => window.seek(t), t);
      await page.screenshot({ path: path.join(dir, `t_${t.toFixed(2).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 88 });
      process.stdout.write(t + ' ');
    }
    console.log();
  } else {
    const fps = +a, t0 = +b, t1 = +c, out = d;
    const n0 = Math.round(t0 * fps), n1 = Math.round(t1 * fps);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps * +(process.env.SUB || 1)), '-c:v', 'mjpeg', '-i', '-',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
    const tStart = Date.now();
    // motion blur: two sub-frames per output frame (180° shutter) when SUB=2
    const SUB = +(process.env.SUB || 1);
    for (let n = n0; n < n1; n++) {
      for (let k = 0; k < SUB; k++) {
        await page.evaluate(t => window.seek(t), n / fps + k / (fps * SUB * 2));
        const buf = await page.screenshot({ type: 'jpeg', quality: 96 });
        if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      }
      if ((n - n0) % 60 === 0) console.log(`frame ${n}/${n1}  ${((Date.now() - tStart) / (n - n0 + 1)).toFixed(0)} ms/frame`);
    }
    ff.stdin.end(); await new Promise(r => ff.on('close', r));
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
