import numpy as np
import imageio.v2 as imageio
import os

W, H = 1920, 1080
os.makedirs("../png", exist_ok=True)

for i in range(60):
    data = np.fromfile(f"../out/img_{i}.data", dtype=np.float32)
    img = data.reshape((H, W, 3))
    img = np.clip(img, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    imageio.imwrite(f"../png/{i:03d}.png", img8)
