"""Remove the printed seconds hand from each watch cut-out so a live, animated
seconds hand can be drawn on top in the ad. The dial pattern is left/right
symmetric, so most of the hidden dial is recovered from its mirror image;
the remainder is inpainted."""
import json, numpy as np, cv2
from PIL import Image

EST = {'black': (176, 294), 'royal': (183, 295), 'teal': (182, 296),
       'ice': (186, 296), 'silver': (182.5, 294)}
out = {}
for name, (cx0, cy0) in EST.items():
    im = np.asarray(Image.open(f'assets/cut/watch_{name}.png').convert('RGBA')).astype(np.float32)
    rgb = im[:, :, :3]
    H, W = rgb.shape[:2]
    # 1) symmetry axis: test mirror axes (2*x0 integer) around estimate
    ys = slice(int(cy0) - 120, int(cy0) + 120)
    best = None
    for twox in range(int(2 * cx0) - 12, int(2 * cx0) + 13):
        xs = np.arange(int(cx0) - 130, int(cx0) + 131)
        xm = twox - xs
        ok = (xm >= 0) & (xm < W)
        d = np.abs(rgb[ys][:, xs[ok]] - rgb[ys][:, xm[ok]]).mean()
        if best is None or d < best[0]:
            best = (d, twox)
    twox = best[1]
    mirror = np.zeros_like(rgb)
    xs = np.arange(W); xm = np.clip(twox - xs, 0, W - 1)
    mirror = rgb[:, xm]
    diff = np.abs(rgb - mirror).sum(axis=2)
    cx = twox / 2
    # 2) seconds-hand angle + centre y: maximise asymmetry along the ray
    bestA = None
    for cy in np.arange(cy0 - 5, cy0 + 5.01, 0.5):
        for ang in np.arange(110, 150, 0.25):  # degrees, image coords (y down), 0=+x
            a = np.radians(ang)
            r = np.arange(35, 110)
            px = cx + r * np.cos(a); py = cy + r * np.sin(a)
            v = diff[np.round(py).astype(int), np.round(px).astype(int)].mean()
            if bestA is None or v > bestA[0]:
                bestA = (v, cy, ang)
    _, cy, ang = bestA
    a = np.radians(ang)
    # 3) tip: walk outward until asymmetry drops
    prof = []
    for r in range(30, 170):
        px = cx + r * np.cos(a); py = cy + r * np.sin(a)
        if not (0 <= px < W and 0 <= py < H): break
        prof.append(diff[int(round(py)), int(round(px))])
    prof = np.convolve(np.array(prof), np.ones(3) / 3, mode='same')
    thr = 0.35 * np.median(prof[5:60])
    tip = 30
    for i, v in enumerate(prof):
        if i > 40 and v < thr: break
        tip = 30 + i
    # 4) mask around the segment
    yy, xx = np.mgrid[0:H, 0:W]
    dx, dy = xx - cx, yy - cy
    along = dx * np.cos(a) + dy * np.sin(a)
    perp = np.abs(-dx * np.sin(a) + dy * np.cos(a))
    seg = (along > 27) & (along < tip + 4) & (perp < 3.2)
    mask = seg & (diff > 25)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool) & (perp < 4.5) & (along > 26)
    res = rgb.copy()
    rnum = tip - 14  # beyond this the hand overlaps the numeral ring: inpaint instead of mirror
    use_mirror = mask & (along < rnum)
    res[use_mirror] = mirror[use_mirror]
    inp = (mask & ~use_mirror).astype(np.uint8)
    res8 = np.clip(res, 0, 255).astype(np.uint8)
    if inp.any():
        res8 = cv2.inpaint(res8, inp * 255, 3, cv2.INPAINT_TELEA)
    o = np.dstack([res8, im[:, :, 3].astype(np.uint8)])
    Image.fromarray(o).save(f'assets/cut/watch_{name}_nosec.png')
    # sample hand colour from original pixels on the hand centreline
    on = seg & (perp < 1.0) & (along > 40) & (along < tip - 10)
    col = rgb[on].mean(axis=0)
    out[name] = dict(cx=round(cx, 2), cy=round(float(cy), 2), angle=round(float(ang), 2), tip=int(tip),
                     color=[int(c) for c in col], w=W, h=H)
    print(name, out[name])
json.dump(out, open('assets/cut/hands.json', 'w'), indent=1)
