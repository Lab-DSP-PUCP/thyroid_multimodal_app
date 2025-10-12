# imports que estás usando pero no aparecen al inicio
import torch #type: ignore
from pathlib import Path
from PIL import Image, ImageDraw #type: ignore
import uuid, io, base64, ast, os
from utils import _count_present_fields
import torchvision.transforms as T #type: ignore
from dotenv import load_dotenv  # type: ignore
from flask import Flask, request, session, redirect, url_for, flash, render_template, send_from_directory, jsonify # type: ignore
import sys

# tus módulos internos (usa el "barrel" que propuse)
from core import (
  ctx,                      # contexto compartido
  allowed_file, load_cat_maps, encode_clinical_with_mask,
  canonicalize_meta, _parse_xml_meta,
  ThyroidAppWrapper, ModelNotReadyError,
  UNetMini, unet_infer_box, overlay_mask_on_full
)
from gradcam_utils import GradCAM, overlay_cam_on_full, overlay_cam_on_pil

# base de datos y blueprint (si los usas)
from db_models import db
from api_history_bp import bp as history_api_bp
from datetime import datetime, timedelta  # en vez de solo timedelta

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
HISTORY_JSON = os.path.join(UPLOAD_FOLDER, "_history.json")
ALLOWED_IMG = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_XML = {"xml"}

# Base y carpeta models (permite override por env var)

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
MODELS_DIR    = Path(os.environ.get("MODELS_DIR", str(BASE_DIR / "models")))


load_dotenv()  # ← lee el .env
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "change-me")

# === ADMIN PASSWORD por entorno (o .env) ===
app.config['ADMIN_PASSWORD'] = os.environ.get('ADMIN_PASSWORD', 'cambialo')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

DB_PATH = BASE_DIR / "thyroid.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


@app.post("/admin/login")
def admin_login():
    pwd = (request.form.get("password") or "").strip()
    ok = bool(pwd) and (pwd == app.config['ADMIN_PASSWORD'])
    if not ok:
        return jsonify({"ok": False, "error": "Credenciales inválidas"}), 401
    session.permanent = True
    session['is_admin'] = True
    return jsonify({"ok": True})

@app.post("/admin/logout")
def admin_logout():
    session.pop('is_admin', None)
    return jsonify({"ok": True})

@app.get("/admin/status")
def admin_status():
    return jsonify({"ok": True, "is_admin": bool(session.get("is_admin"))})

db.init_app(app)
with app.app_context():
    db.create_all()  # en prod usa migraciones (Flask-Migrate)
app.register_blueprint(history_api_bp)

# Rutas por defect
WEIGHTS      = os.environ.get("WEIGHTS_PATH",   str(MODELS_DIR / "ResNet18_ensemble_holdout.pth"))
CAT_MAPS_PATH= os.environ.get("CAT_MAPS_PATH",  str(MODELS_DIR / "cat_maps.json"))
UNET_PTH     = os.environ.get("UNET_PTH",       str(MODELS_DIR / "unet_best.pth"))

# Cargar U-Net (fallback ROI)
UNET = None
if Path(UNET_PTH).exists():
    UNET = UNetMini(in_ch=1, out_ch=1, base=16, depth=4).to(DEVICE).eval()
    state = torch.load(UNET_PTH, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    UNET.load_state_dict(state, strict=True)
    app.logger.info(f"[U-Net] cargada desde {UNET_PTH}")
else:
    app.logger.warning(f"[U-Net] no encontrada en {UNET_PTH}; fallback a center-crop cuando falte ROI.")

# 1) Intenta leer cat_maps desde el checkpoint (si lo trae)
ck_cat_maps = None
try:
    _ck = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(_ck, dict) and "cat_maps" in _ck:
        ck_cat_maps = _ck["cat_maps"]
except Exception:
    pass  # si falla, sigue con el JSON

# 2) Lee cat_maps del JSON de disco
file_cat_maps = load_cat_maps(CAT_MAPS_PATH)

SENTINELS = {"", "-", "none", "null", "n/a"}

# 3) Decide el mapping definitivo: prioriza el del checkpoint si existe
def _canon(m):  # para comparar por orden de claves (define el one-hot)
    return {
        k: list(m[k].keys()) if isinstance(m[k], dict) else list(m[k])
        for k in ["composition","echogenicity","margins","calcifications","sex"]
    }

if ck_cat_maps is not None and _canon(ck_cat_maps) != _canon(file_cat_maps):
    print("⚠️ cat_maps.json difiere del del checkpoint. Usaré el del checkpoint.")
    cat_maps = ck_cat_maps
else:
    cat_maps = file_cat_maps

# 4) Dimension tabular con el mapping definitivo
tabular_input_dim = (
    len(cat_maps["composition"]) +
    len(cat_maps["echogenicity"]) +
    len(cat_maps["margins"]) +
    len(cat_maps["calcifications"]) +
    len(cat_maps["sex"]) + 1  # + age
)

# Etiquetas de UI (titulos de columnas / campos)
UI_LABELS = {
    "composition": "Composición",
    "echogenicity": "Ecogenicidad",
    "margins": "Márgenes",
    "calcifications": "Calcificaciones",
    "sex": "Sexo",
    "age": "Edad",
}

# Etiquetas legibles para los valores (mapeo a espanhol)
VALUE_LABELS = {
    "composition": {"solid": "Sólido", "mixed": "Mixto", "cystic": "Quístico"},
    "echogenicity": {
        "hypoechoic": "Hipoecoico",
        "isoechoic": "Isoecoico",
        "hyperechogenicity": "Hiperecoico",
        "hyperechoic": "Hiperecoico",
        "very_hypoechoic": "Muy hipoecoico",
    },
    "margins": {
        "smooth": "Lisos",
        "ill_defined": "Mal definidos",
        "irregular": "Irregulares",
        "extrathyroidal": "Ext. tiroidea",
        "well defined": "Bien definidos",
        "well_defined": "Bien definidos",
    },
    "calcifications": {
        "none": "Ninguna",
        "non": "Ninguna",
        "micro": "Micro",
        "macro": "Macro",
        "peripheral": "Periféricas",
    },
    "sex": {"F": "Femenino", "M": "Masculino"},
}

ROI_LABELS = {
    "xml":    "XML",
    "unet":   "U-Net",
    "meta":   "Metadatos",
    "center": "-",   # cuando no hubo ROI y usa center-crop
    "-":      "-",   # clave de seguridad para “sin dato”
}

# 5) Instancia el wrapper con la dimensión correcta
try:
    wrapper = ThyroidAppWrapper(weights_path=WEIGHTS, tabular_input_dim=tabular_input_dim)
    model_ready = True
except ModelNotReadyError:
    wrapper = None
    model_ready = False

TARGET_LAYER = wrapper.model.cnn.layer4[-1].conv2 if wrapper else None

def load_history():
    import json
    if not os.path.exists(HISTORY_JSON):
        return []
    try:
        return json.load(open(HISTORY_JSON, "r", encoding="utf-8"))
    except Exception:
        return []

def save_history(history):
    import json
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    json.dump(history, open(HISTORY_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

@app.route("/", methods=["GET"])
def index():
    history = [h for h in load_history() if not h.get("deleted_at")]

    # --- Normalización de ROI para la UI (corrige historiales viejos/meta/manual) ---
    for h in history:
        src = str(h.get("roi_source") or "").strip().lower()
        # Corrige etiquetas antiguas o faltantes
        if src in ("xml","unet","center","-"):
            pass
        elif src == "meta":      # compat: 'meta' debe verse como XML si hay box y XML
            if h.get("xml") and h["xml"] != "-" and h.get("box") and h["box"] != "-":
                src = "xml"
            elif h.get("box") and h["box"] != "-":
                src = "unet"
            else:
                src = "center"
        elif src == "manual":    # compat: sin XML, con box -> unet; sin box -> center
            src = "unet" if (h.get("box") and h["box"] != "-") else "center"
        else:
            # Deducción por presencia de XML/BOX si roi_source está vacío
            if h.get("xml") and h["xml"] != "-" and h.get("box") and h["box"] != "-":
                src = "xml"
            elif h.get("box") and h["box"] != "-":
                src = "unet"
            else:
                src = "center"
        h["roi_source"] = src
    values_map = {
        "composition": {
            0: "Quística",
            1: "Espongiforme",
            2: "Mixta",
            3: "Sólida",
        },
        "echogenicity": {
            0: "Muy hipoecoica",
            1: "Hipoecoica",
            2: "Isoecoica",
            3: "Hiperecoica",
        },
        "margins": {
            0: "Definidos",
            1: "Indefinidos",
            2: "Irregulares",
            3: "Extensión extratiroidea",
        },
        "calcifications": {
            0: "Ninguna",
            1: "Macrocalcificaciones",
            2: "Puntiformes",
            3: "Sombra osificante",
        },
        "sex": {
            0: "Femenino",
            1: "Masculino",
        },
    }
    return render_template(
        "index.html",
        history=history,
        model_ready=model_ready,
        cat_maps=cat_maps,
        ui_labels=UI_LABELS,
        value_labels=VALUE_LABELS,
        values_map=values_map,
        roi_labels=ROI_LABELS
    )

# Mismos transforms del modelo
to_tensor = T.ToTensor()
norm_L    = T.Normalize([0.5],[0.5])

def _parse_box_maybe(box, W, H):
    if box is None: return None
    if isinstance(box, str):
        box = box.strip()
        if not box or box.lower() in ('nan','none','null'):
            return None
        try: box = ast.literal_eval(box)
        except Exception: return None
    if not hasattr(box,'__len__') or len(box)!=4: return None
    x0,y0,x1,y1 = map(float, box)
    if max(x0,x1,y0,y1) <= 1.0: x0,x1 = x0*W, x1*W; y0,y1 = y0*H, y1*H
    x0 = max(0, min(int(round(x0)), W-1))
    y0 = max(0, min(int(round(y0)), H-1))
    x1 = max(x0+1, min(int(round(x1)), W))
    y1 = max(y0+1, min(int(round(y1)), H))
    if (x1-x0)<5 or (y1-y0)<5: return None
    return (x0,y0,x1,y1)

def resize_center_crop_pil(pil_img, resize_to=256, crop_size=224):
    w,h = pil_img.size
    if w<=h:
        new_w,new_h = resize_to, int(h*resize_to/w)
    else:
        new_h,new_w = resize_to, int(w*resize_to/h)
    img_res = pil_img.resize((new_w,new_h), Image.BILINEAR)
    left=(new_w-crop_size)//2; top=(new_h-crop_size)//2
    return img_res.crop((left,top,left+crop_size,top+crop_size))

def _prepare_image_with_roi(img_path, xml_path=None):
    """
    Devuelve:
      x_img   : Tensor [1,3,224,224] listo para el modelo
      roi_src : 'meta' | 'unet' | 'center'
      box     : (x0,y0,x1,y1) o None (en coords de la imagen original)
      full_rgb: PIL RGB de la imagen completa
      crop_rgb: PIL RGB del recorte usado
    """
    # 1) Abrir imagen completa en L (como en notebook) y RGB para overlay
    full_L   = Image.open(img_path).convert('L')
    full_rgb = Image.open(img_path).convert('RGB')
    W, H     = full_L.size

    # 2) ROI desde XML (si existe y tiene roi/box normalizado o en pixeles)
    box_meta = None
    if xml_path:
        try:
            xml_meta_raw  = _parse_xml_meta(xml_path)
            xml_meta_norm = canonicalize_meta(xml_meta_raw, cat_maps)
            # acepta 'roi' o 'box' en el XML normalizado
            box_meta = xml_meta_norm.get('roi', None) or xml_meta_norm.get('box', None)
            box_meta = _parse_box_maybe(box_meta, W, H)
        except Exception:
            box_meta = None

    # 3) fallback: U-Net
    box = box_meta
    roi_src = 'xml' if box is not None else None
    if box is None and (UNET is not None):
        try:
            box_pred, _, _ = unet_infer_box(UNET, full_L, device=DEVICE, min_area=80, margin=0.10)
            box = _parse_box_maybe(box_pred, W, H)
            if box is not None:
                roi_src = 'unet'
        except Exception:
            box = None

    # 4) Ultimo fallback: center
    if box is None:
        roi_src = 'center'
        crop_rgb = full_rgb
    else:
        crop_rgb = full_rgb.crop(box)

    # 5) Resize(256)+CenterCrop(224)+ToTensor+Normalize
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),  # usa tus stats si difieren
    ])
    x_img = tfm(crop_rgb).unsqueeze(0).to(wrapper.device if wrapper else DEVICE)

    return x_img, roi_src, box, full_rgb, crop_rgb

ctx.bind(
    wrapper_obj=wrapper,
    cat_maps_obj=cat_maps,
    device_obj=DEVICE,
    prepare_image_fn=_prepare_image_with_roi,
    encode_clinical_fn=encode_clinical_with_mask
)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        flash("Sube una imagen de ecografía.", "warning")
        return redirect(url_for('index'))

    image_file = request.files['image']
    if not allowed_file(image_file.filename, ALLOWED_IMG):
        flash("Formato de imagen no permitido.", "warning")
        return redirect(url_for('index'))

    # Guarda imagen
    uid = uuid.uuid4().hex[:8]
    filename = image_file.filename.replace(" ", "_")
    saved_img = f"{uid}_{filename}"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_img)
    image_file.save(img_path)

    # Guarda XML (opcional)
    xml_path = None
    if 'xmlfile' in request.files and request.files['xmlfile'].filename not in ("", None):
        xmlf = request.files['xmlfile']
        if allowed_file(xmlf.filename, ALLOWED_XML):
            saved_xml = f"{uid}_{xmlf.filename.replace(' ', '_')}"
            xml_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_xml)
            xmlf.save(xml_path)

    # Metadata del form + complementos del XML si existe
    clinical_payload = {
        "composition": request.form.get("composition","") or "",
        "echogenicity": request.form.get("echogenicity","") or "",
        "margins": request.form.get("margins","") or "",
        "calcifications": request.form.get("calcifications","") or "",
        "sex": request.form.get("sex","") or "",
        "age": request.form.get("age","") or "",
    }

    # Si hay XML: parsea profundo + normaliza (keys/valores) y solo rellena vacios del form
    if xml_path:
        xml_meta_raw  = _parse_xml_meta(xml_path)
        xml_meta_norm = canonicalize_meta(xml_meta_raw, cat_maps)
        for k in ["composition","echogenicity","margins","calcifications","sex","age"]:
            if not (clinical_payload.get(k) or "").strip():
                v = xml_meta_norm.get(k)
                if v:
                    clinical_payload[k] = v

    k_present = _count_present_fields(clinical_payload)
    coverage = k_present / 6.0
    meta_used_flag = k_present > 0

    # Imagen + ROI (meta -> U-Net -> center) y tensor final
    try:
        img_tensor, roi_source, box, full_rgb, crop_rgb = _prepare_image_with_roi(img_path, xml_path)
    except Exception as e:
        flash(f"Error al preparar imagen/ROI: {e}", "warning")
        return redirect(url_for('index'))
    # Asegura etiqueta coherente para guardar y para la UI
    if xml_path and box:
        roi_source = "xml"
    elif box:
        roi_source = "unet"
    else:
        roi_source = "center"

    # Vector clinico + cobertura + k
    vec, coverage, k_present = encode_clinical_with_mask(clinical_payload, cat_maps)
    clin_tensor = torch.tensor(vec, dtype=torch.float32) if any(vec) else None
    if clin_tensor is not None:
        clin_tensor = clin_tensor.unsqueeze(0).to(wrapper.device)

    # Prediccion
    if not model_ready:
        flash("Modelo no cargado.", "warning")
        return redirect(url_for('index'))

    with wrapper.inference_mode():
        _tmp_pred, prob = wrapper.predict(img_tensor, clin_tensor)

    # Seleccion de tau
    tau_img = getattr(wrapper, "tau_img", getattr(wrapper, "tau_mm", 0.5))
    tau_mm  = getattr(wrapper, "tau_mm", 0.5)
    tau_by_k = getattr(wrapper, "tau_by_k", {})

    if isinstance(k_present, (int, float)) and int(k_present) in tau_by_k:
        tau = float(tau_by_k[int(k_present)])
    else:
        # Fallback: mezcla lineal por cobertura
        tau = tau_img + coverage * (tau_mm - tau_img)

    pred  = 1 if float(prob) >= float(tau) else 0
    label = "Maligno" if pred == 1 else "Benigno"

    patient_id = (request.form.get("patient_id","") or "").strip()
    if not patient_id:
        flash("El ID de paciente es obligatorio.", "warning")
        return redirect(url_for('index'))

    result = {
        "id": uid,
        "patient_id": patient_id,
        "file": saved_img,
        "xml": os.path.basename(xml_path) if xml_path else "-",
        "label": label,
        "pred": int(pred),
        "prob": round(float(prob), 4),
        "meta_used": meta_used_flag,
        "meta_coverage": round(float(coverage), 3),
        "k_present": int(k_present),
        "composition": clinical_payload.get("composition") or "-",
        "echogenicity": clinical_payload.get("echogenicity") or "-",
        "margins": clinical_payload.get("margins") or "-",
        "calcifications": clinical_payload.get("calcifications") or "-",
        "sex": clinical_payload.get("sex") or "-",
        "age": clinical_payload.get("age") or "-",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "roi_source": roi_source or "-",
        "box": box if box else "-",
    }

    # Guarda una imagen con el recuadro para previsualizacion rapida
    try:
        overlay_roi = full_rgb.copy()
        if box:
            draw = ImageDraw.Draw(overlay_roi)
            color = (0,255,0) if roi_source=='xml' else (255,165,0)
            draw.rectangle(box, outline=color, width=4)
        preview_name = f"{uid}_roi.png"
        overlay_roi.save(os.path.join(app.config['UPLOAD_FOLDER'], preview_name))
        result["roi_preview"] = preview_name
    except Exception:
        result["roi_preview"] = "-"

    history = load_history()
    history.insert(0, result)
    save_history(history[:50])

    return redirect(url_for('index'))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/explain", methods=["POST"])
def explain():
    if not model_ready or wrapper is None:
        return {"error": "Modelo no cargado"}, 503

    # Entrada
    f        = request.files.get('file', None)
    filename = (request.form.get('filename', '') or '').strip()

    if f is None and not filename:
        return {"error": "Falta 'file' o 'filename'."}, 400

    # Intenta recuperar el XML asociado si viene por filename (para usar MISMA ROI)
    img_path = None
    xml_path_for_explain = None
    if not f and filename:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(img_path):
            return {"error": "No existe el archivo solicitado."}, 404
        try:
            hist = load_history()
            rec  = next((h for h in hist if h.get("file") == filename), None)
            if rec and rec.get("xml") and rec["xml"] != "-":
                xml_path_for_explain = os.path.join(app.config['UPLOAD_FOLDER'], rec["xml"])
        except Exception:
            xml_path_for_explain = None

    # Prepara tensor de imagen + ROI consistente (meta -> UNet -> center)
    # Para 'file' subido por stream, guarda temporalmente para reutilizar el helper.
    tmp_path = None
    try:
        if img_path:  # por filename
            x, roi_source, box, full_rgb, crop_rgb = _prepare_image_with_roi(img_path, xml_path_for_explain)
        else:
            tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"tmp_{uuid.uuid4().hex[:8]}.png")
            Image.open(f.stream).convert('RGB').save(tmp_path, format='PNG')
            x, roi_source, box, full_rgb, crop_rgb = _prepare_image_with_roi(tmp_path, None)
    except Exception as e:
        if tmp_path:
            try: os.remove(tmp_path)
            except: pass
        return {"error": f"Preprocesamiento/ROI falló: {e}"}, 500
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except: pass

    # Clinicos opcionales
    # Metadatos clinicos: si llegan, se incluyen siempre
    clin = None
    try:
        payload = {
            "composition":   request.form.get("composition",""),
            "echogenicity":  request.form.get("echogenicity",""),
            "margins":       request.form.get("margins",""),
            "calcifications":request.form.get("calcifications",""),
            "sex":           request.form.get("sex",""),
            "age":           request.form.get("age",""),
        }
        if any((payload or {}).values()):
            vec, _ = encode_clinical_with_mask(payload, cat_maps)
            clin = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(wrapper.device)
    except Exception:
        clin = None  # si algo falla, no bloquea el CAM

    # Grad-CAM sobre el mismo layer definido en TARGET_LAYER
    model = wrapper.model.to(wrapper.device).eval()
    if TARGET_LAYER is None:
        return {"error": "TARGET_LAYER no disponible."}, 500

    # Proxy para pasar clinicos fijos (el modelo espera forward(x_img, x_tab))
    class _Proxy(torch.nn.Module):
        def __init__(self, base, clin_tensor):
            super().__init__()
            self.base = base
            self.clin = clin_tensor
        def forward(self, x):
            return self.base(x, self.clin)

    proxy = _Proxy(model, clin).to(wrapper.device).eval()
    cam_engine = GradCAM(proxy, TARGET_LAYER)

    try:
        with torch.enable_grad():
            cam, used_idx = cam_engine(x, class_idx=None)
    finally:
        cam_engine.remove_hooks()

    # Overlay (elige vista): 'crop' = lo que vio el modelo | 'full' = CAM en imagen completa
    alpha = float(request.form.get('alpha', 0.35))
    view  = (request.form.get('view', 'crop') or 'crop').lower()  # 'crop' | 'full'

    if view == 'full' and box:
        overlay = overlay_cam_on_full(full_rgb, cam, box, alpha=alpha)
    else:
        # si no hay box o se pidio 'crop', muestra el CAM sobre el recorte
        overlay = overlay_cam_on_pil(crop_rgb, cam, alpha=alpha)

    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        "overlay_png_base64": f"data:image/png;base64,{b64}",
        "used_class_idx": int(used_idx),
        "roi_source": roi_source or "-",
        "box": box if box else None,
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)