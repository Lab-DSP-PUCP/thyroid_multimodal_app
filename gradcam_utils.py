import torch # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import matplotlib.cm as cm # type: ignore

class GradCAM:
    """Simple Grad-CAM for PyTorch (works with ResNet-18 last conv)."""
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        try: self.fwd_hook.remove()
        except Exception: pass
        try: self.bwd_hook.remove()
        except Exception: pass

    @torch.no_grad()
    def _normalize_cam(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def __call__(self, input_tensor, class_idx=None):
        """Returns (cam_np[H,W] in [0,1], used_class_idx)."""
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)

            if logits.dim() == 2 and logits.shape[1] == 1:  # binary
                score = logits[:, 0]; used_idx = 0
            else:
                used_idx = int(logits.argmax(dim=1).item()) if class_idx is None else int(class_idx)
                score = logits[:, used_idx]

            score.backward(retain_graph=True)
            acts  = self.activations          # (B,K,h,w)
            grads = self.gradients            # (B,K,h,w)
            w = grads.mean(dim=(2,3), keepdim=True)       # (B,K,1,1)
            cam = (w * acts).sum(dim=1, keepdim=True)     # (B,1,h,w)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
            cam = cam[0,0].cpu().numpy()
            cam = self._normalize_cam(cam)
            return cam, used_idx

def overlay_cam_on_pil(pil_img, cam, alpha=0.35, cmap_name='jet'):
    """Overlay CAM heatmap over a PIL image (RGB)."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    w, h = pil_img.size
    cam_img = Image.fromarray((cam*255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
    cam_np = np.array(cam_img) / 255.0
    cmap = cm.get_cmap(cmap_name)
    heat = cmap(cam_np)[..., :3]   # RGB in [0,1]
    base = np.array(pil_img).astype(np.float32) / 255.0
    over = (1 - alpha) * base + alpha * heat
    over = np.clip(over, 0, 1)
    return Image.fromarray((over * 255).astype(np.uint8))

def overlay_cam_on_full(full_rgb, cam, box, alpha=0.35, cmap_name='jet'):
    """
    Pega el CAM (computado sobre el RECORTE) dentro de la imagen completa.
    - full_rgb: PIL RGB de la imagen completa.
    - cam     : np.float32 [0..1] con tamaño del recorte (p.ej. 224x224).
    - box     : (x0,y0,x1,y1) en coords del full frame.
    """
    if full_rgb.mode != 'RGB':
        full_rgb = full_rgb.convert('RGB')
    x0, y0, x1, y1 = map(int, box)
    crop = full_rgb.crop((x0, y0, x1, y1))
    over_crop = overlay_cam_on_pil(crop, cam, alpha=alpha, cmap_name=cmap_name)
    out = full_rgb.copy()
    out.paste(over_crop, (x0, y0))
    return out