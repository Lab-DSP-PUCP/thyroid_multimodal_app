import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
import numpy as np #type: ignore
from PIL import Image, ImageOps #type: ignore
import torchvision.transforms as T #type: ignore
import cv2 #type: ignore
from matplotlib import cm as mpl_cm #type: ignore

# U-Net mini (ligera)
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class UNetMini(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16, depth=4):
        super().__init__()
        self.depth = depth
        chs = [base * (2**i) for i in range(depth)]

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_ch
        for c in chs:
            self.downs.append(conv_block(c_in, c))
            self.pools.append(nn.MaxPool2d(2))
            c_in = c

        self.bottleneck = conv_block(chs[-1], chs[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.ups = nn.ModuleList()
        c_in = chs[-1] * 2
        for c in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(c_in, c, 2, 2))
            self.ups.append(conv_block(c * 2, c))
            c_in = c

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.downs[i](x)
            skips.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skips[self.depth - 1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i](x)

        return self.head(x)  # logits (sin sigmoid)

# Letterbox + back-mapping
def letterbox_resize(pil_img, out_size=(512, 512), fill=0):
    W, H = pil_img.size
    out_w, out_h = out_size
    scale = min(out_w / W, out_h / H)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    img_res = pil_img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = out_w - new_w
    pad_h = out_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    img_pad = ImageOps.expand(
        img_res,
        border=(pad_left, pad_top, pad_w - pad_left, pad_h - pad_top),
        fill=fill,
    )

    meta = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "orig_w": W,
        "orig_h": H,
        "out_w": out_w,
        "out_h": out_h,
    }
    return img_pad, meta


def map_box_back(box_lb, meta):
    x0, y0, x1, y1 = box_lb
    sc = meta["scale"]
    pl = meta["pad_left"]
    pt = meta["pad_top"]

    x0 = (x0 - pl) / sc
    x1 = (x1 - pl) / sc
    y0 = (y0 - pt) / sc
    y1 = (y1 - pt) / sc

    x0 = max(0, min(int(round(x0)), meta["orig_w"] - 1))
    y0 = max(0, min(int(round(y0)), meta["orig_h"] - 1))
    x1 = max(x0 + 1, min(int(round(x1)), meta["orig_w"]))
    y1 = max(y0 + 1, min(int(round(y1)), meta["orig_h"]))
    return (x0, y0, x1, y1)

# Post-proc: mascara -> bbox
def get_box_from_mask(mask_np, min_area=50, margin=0.10):
    try:
        H, W = mask_np.shape
        bin_img = (mask_np >= 0.5).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
        best = None
        for i in range(1, num):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if best is None or a > best[-1]:
                best = (x, y, x + w, y + h, a)
        if best is None:
            return None
        x0, y0, x1, y1, _ = best
    except Exception:
        pos = np.argwhere(mask_np >= 0.5)
        if pos.size == 0:
            return None
        y0, x0 = pos.min(axis=0)
        y1, x1 = pos.max(axis=0) + 1

    dx = int(round((x1 - x0) * margin))
    dy = int(round((y1 - y0) * margin))
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = x1 + dx
    y1 = y1 + dy
    return (int(x0), int(y0), int(x1), int(y1))

# Inferencia U-Net
SEG_SIZE = 512
_seg_tfm = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

@torch.no_grad()
def unet_infer_box(unet, pil_img_L, device=None, min_area=50, margin=0.10):
    device = device or next(unet.parameters()).device
    lb_img, meta = letterbox_resize(pil_img_L, (SEG_SIZE, SEG_SIZE), fill=0)
    x = _seg_tfm(lb_img).unsqueeze(0).to(device)  # [1,1,H,W]
    logits = unet(x)                               # [1,1,H,W]
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    box_lb = get_box_from_mask(prob, min_area=min_area, margin=margin)
    if box_lb is None:
        return None, prob, meta
    box_orig = map_box_back(box_lb, meta)
    return box_orig, prob, meta

# Overlay de mascara en full-frame
def overlay_mask_on_full(full_img, mask_lb, meta, alpha=0.35):
    H_lb, W_lb = mask_lb.shape
    ow, oh = meta["out_w"], meta["out_h"]
    pl, pt = meta["pad_left"], meta["pad_top"]

    heat_rgb = (mpl_cm.get_cmap('jet')(mask_lb)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb).resize((ow, oh), Image.BILINEAR)

    crop = heat_img.crop((pl, pt, pl + int(round(W_lb)), pt + int(round(H_lb))))
    mapped = crop.resize((meta["orig_w"], meta["orig_h"]), Image.BILINEAR)

    base = full_img.convert("RGB").copy()
    return Image.blend(base, mapped, alpha=alpha)