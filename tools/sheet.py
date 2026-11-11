import sys, glob
from PIL import Image
d, prefix = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(d + '/t_*.jpg'))
ims = [Image.open(f).resize((640, 360)) for f in fs]
for k in range(0, len(ims), 6):
    grp = ims[k:k + 6]
    s = Image.new('RGB', (1280, 360 * ((len(grp) + 1) // 2)))
    for i, im in enumerate(grp): s.paste(im, ((i % 2) * 640, (i // 2) * 360))
    s.save(f'{prefix}_{k // 6}.jpg', quality=88)
print(len(fs), 'stills')
