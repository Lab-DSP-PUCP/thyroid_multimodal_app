
# Grad-CAM para tu proyecto (ResNet18 + rama tabular)

A continuación tienes **archivos listos** y **snippets** para integrar Grad-CAM en tu **notebook** y en tu **app Flask**.

---

## 1) Archivo utilitario

Copia `gradcam_utils.py` a la carpeta del proyecto:
```
thyroid_multimodal_app/
  app.py
  model.py
  utils.py
  gradcam_utils.py   <--- (nuevo)
  static/...
  uploads/...
```

---

## 2) Notebook — Sección “Explainability (Grad‑CAM)”

**Colócalo inmediatamente después** de donde instancias tu modelo (o `ThyroidAppWrapper`).

```python
from PIL import Image
import torch
from gradcam_utils import GradCAM, overlay_cam_on_pil

# 1) Obtener el modelo de imagen (la rama CNN) desde tu wrapper
device = wrapper.device  # o: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = wrapper.model.to(device).eval()

# 2) Capa objetivo en ResNet-18 (última conv):
target_layer = model.cnn.layer4[-1].conv2

# 3) Instanciar Grad-CAM
cam_engine = GradCAM(model, target_layer)

# 4) Preprocesar una eco (usa tu mismo pipeline que en la app)
img_path = 'thyroid_multimodal_app/uploads/5e7b09ca_2_1.jpg'  # cambia a tu imagen
x = utils.preprocess_image(img_path)     # [3,224,224]
x = x.unsqueeze(0).to(device)            # [1,3,224,224]

# Si quieres incluir metadatos tabulares para que el gradiente sea con el mismo contexto:
# clin = utils.encode_clinical(payload, cat_maps)  # vector 1D
# clin = torch.tensor(clin, dtype=torch.float32).unsqueeze(0).to(device)

cam, used_idx = cam_engine(x, class_idx=None)  # para binario, used_idx=0
overlay = overlay_cam_on_pil(Image.open(img_path).convert('RGB'), cam, alpha=0.35)
overlay.save('gradcam_overlay_demo.png')
cam_engine.remove_hooks()

print('Clase usada:', used_idx)
```

**Notas rápidas**
- Para ResNet‑18 la capa recomendada es `model.cnn.layer4[-1].conv2`.
- Para ResNet‑50 sería `model.cnn.layer4[-1].conv3`.
- Si tu output es binario `[B,1]`, Grad‑CAM usa el logit único (índice 0).

---

## 3) App Flask — Endpoint `/explain`

### 3.1. Importa y define la capa objetivo

En `app.py`, junto con tus otros imports:

```python
from gradcam_utils import GradCAM, overlay_cam_on_pil
```

Después de crear el `wrapper` (busca donde haces `wrapper = ThyroidAppWrapper(...)`), añade:

```python
# Capa objetivo (ResNet-18)
TARGET_LAYER = wrapper.model.cnn.layer4[-1].conv2 if wrapper else None
```

### 3.2. Endpoint nuevo

Añade esto **antes** del `if __name__ == "__main__":`:

```python
import io, base64
from PIL import Image
import torch

@app.route("/explain", methods=["POST"])
def explain():
    if not model_ready or wrapper is None:
        return {"error": "Modelo no cargado"}, 503

    # Entrada: o un archivo 'file' o un nombre 'filename' de la carpeta uploads/
    f = request.files.get('file', None)
    filename = request.form.get('filename', '').strip()

    if f is None and not filename:
        return {"error":"Falta 'file' o 'filename'."}, 400

    if f:
        img = Image.open(f.stream).convert('RGB')
    else:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return {"error":"No existe el archivo solicitado."}, 404
        img = Image.open(img_path).convert('RGB')

    # Preprocesamiento idéntico al de inferencia
    # (reutilizamos la función de utils que ya normaliza a [3,224,224])
    # Guardamos temporalmente si vino por stream para pasar ruta, o procesamos directo:
    try:
        # Si prefieres evitar tocar disco, puedes duplicar la lógica de utils.preprocess_image aquí.
        tmp = io.BytesIO()
        img.save(tmp, format='PNG'); tmp.seek(0)
        # Trick: PIL -> tensor con tu mismo pipeline
        # Como utils.preprocess_image requiere ruta, reescribirlo a memoria implicaría adaptarlo.
        # Alternativa directa: replicar el _img_tfms aquí. Para mantenerlo simple:
        from utils import _IMAGENET_MEAN, _IMAGENET_STD
        from torchvision import transforms as T
        tfm = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        x = tfm(img).unsqueeze(0).to(wrapper.device)
    except Exception as e:
        return {"error": f"Preprocesamiento falló: {e}"}, 500

    # Metadatos clínicos opcionales (si te interesa que el gradiente sea con el mismo contexto)
    # Recibimos el payload opcional en el POST y lo codificamos cuando esté presente
    clin = None
    try:
        if request.form.get('use_clinical', '0') == '1':
            from utils import encode_clinical_with_mask, load_cat_maps
            cat_maps = load_cat_maps(os.path.join(os.path.dirname(__file__), "cat_maps.json"))
            payload = {
                "composition": request.form.get("composition",""),
                "echogenicity": request.form.get("echogenicity",""),
                "margins": request.form.get("margins",""),
                "calcifications": request.form.get("calcifications",""),
                "sex": request.form.get("sex",""),
                "age": request.form.get("age",""),
            }
            clin_vec, _ = encode_clinical_with_mask(payload, cat_maps)
            clin = torch.tensor(clin_vec, dtype=torch.float32).unsqueeze(0).to(wrapper.device)
    except Exception:
        clin = None  # No bloqueamos por esto

    # Grad-CAM por request (para evitar problemas de hooks concurrentes)
    model = wrapper.model.to(wrapper.device).eval()
    if TARGET_LAYER is None:
        return {"error":"TARGET_LAYER no disponible."}, 500

    cam_engine = GradCAM(model, TARGET_LAYER)
    try:
        # Construimos la entrada que espera tu modelo (imagen + clínico opcional)
        # Grad-CAM internamente hará el forward y backward sobre los logits
        with torch.enable_grad():
            # El modelo original recibe (img, clin) en forward
            # Creamos un pequeño wrapper anónimo que matchee la firma original
            class _Proxy(torch.nn.Module):
                def __init__(self, base, clin):
                    super().__init__()
                    self.base = base
                    self.clin = clin
                def forward(self, x):
                    return self.base(x, self.clin)

            proxy = _Proxy(model, clin).to(wrapper.device).eval()
            cam_engine.model = proxy  # sustituimos el modelo por el proxy con clin fijo

            cam, used_idx = cam_engine(x, class_idx=None)
    finally:
        cam_engine.remove_hooks()

    # Overlay
    from gradcam_utils import overlay_cam_on_pil
    overlay = overlay_cam_on_pil(img, cam, alpha=float(request.form.get('alpha', 0.35)))

    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {
        "overlay_png_base64": f"data:image/png;base64,{b64}",
        "used_class_idx": int(used_idx),
    }
```

### 3.3. Frontend mínimo (botón)

En tu HTML/JS (donde ya manejas la subida de imagen), añade un botón y un `<img>` para mostrar el overlay:

```html
<button id="btn-explain" type="button">Explicar predicción (Grad‑CAM)</button>
<img id="gradcam-view" style="display:none;max-width:100%;border-radius:12px;margin-top:8px;"/>
```

Y en tu JS:

```js
const btnExplain = document.getElementById('btn-explain');
const gradcamView = document.getElementById('gradcam-view');

btnExplain?.addEventListener('click', async () => {
  const fileInput = document.getElementById('file'); // tu input actual
  if (!fileInput || !fileInput.files || !fileInput.files.length) {
    alert('Sube una imagen primero.'); return;
  }
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  // Si quieres incluir clínico:
  // fd.append('use_clinical', '1');
  // fd.append('composition', ...); fd.append('age', ...); etc.

  const res = await fetch('/explain', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.overlay_png_base64) {
    gradcamView.src = data.overlay_png_base64;
    gradcamView.style.display = 'block';
    gradcamView.scrollIntoView({behavior:'smooth', block:'center'});
  } else {
    alert('No se pudo generar el Grad‑CAM.');
  }
});
```

---

## 4) Recomendaciones de uso
- Ejecuta Grad‑CAM **solo bajo demanda** (botón) para no impactar la latencia.
- Si agregas *batching* o *multi‑request*, instancia `GradCAM` **por request** (como arriba) y recuerda llamar `remove_hooks()`.
- Para casos binarios `[B,1]` el CAM usa el **logit positivo** (índice 0). Si migras a 2 salidas `[B,2]`, podrás pasar `class_idx=0/1` explícitamente.