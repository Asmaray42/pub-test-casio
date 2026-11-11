"""4x EDSR super-resolution of the watch cut-outs (colour via EDSR, alpha via Lanczos)."""
import cv2, sys, numpy as np
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(sys.argv[1]); sr.setModel('edsr', 4)
for name in sys.argv[2:]:
    img = cv2.imread(f'assets/cut/watch_{name}_nosec.png', cv2.IMREAD_UNCHANGED)
    rgb = sr.upsample(np.ascontiguousarray(img[:, :, :3]))
    a = cv2.resize(img[:, :, 3], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    # gentle unsharp mask to lift micro-detail of the brushed steel
    blur = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    rgb = cv2.addWeighted(rgb, 1.35, blur, -0.35, 0)
    out = np.dstack([rgb, a])
    # downsample 4x -> 3x for a good balance between size and detail
    out = cv2.resize(out, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'ad/img/watch_{name}.png', out)
    print(name, out.shape, flush=True)
