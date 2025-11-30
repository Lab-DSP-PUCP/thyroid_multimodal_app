import ast
import os
import json
import datetime
from PIL import Image # type: ignore
import torch # type: ignore
from torchvision import transforms as T # type: ignore
from xml.etree import ElementTree as ET
from flask import current_app, abort  # type: ignore
import app_context as ctx

# Normalizacion de metadatos desde XML / texto libre 
_FIELD_ALIASES = {
    "composition":  {"composition","composicion","composición","structure","tipo","composite"},
    "echogenicity": {"echogenicity","ecogenicidad","echogenicidad","eco","echog","echogen"},
    "margins":      {"margins","margin","margen","margenes","márgenes","bordes"},
    "calcifications":{"calcifications","calcificacion","calcificaciones","calc","microcalcifications","macrocalcifications","peripheral_calcifications"},
    "sex":          {"sex","sexo","gender"},
    "age":          {"age","edad","years","age_years","edad_años"},
}

# Sinonimos comunes -> key canonica del cat_maps
_VALUE_ALIASES = {
    "composition": {
        "sólido":"solid","solido":"solid","solid":"solid",
        "mixto":"mixed","mixed":"mixed",
        "quístico":"cystic","quistico":"cystic","cystic":"cystic",
        "esponjiforme":"spongiform","espongiforme":"spongiform",
        "spongiforme":"spongiform","spongiform":"spongiform",
        "dense":"dense","denso":"dense",
        "predominantly solid":"predominantly solid",
        "predominantly_solid":"predominantly solid",
        "predominantly cystic":"predominantly cystic",
        "predominantly_cystic":"predominantly cystic",
    },
    "echogenicity": {
        "hypoechoic":"hypoechogenicity",
        "isoechoic":"isoechogenicity",
        "hyperechoic":"hyperechogenicity",
        "hipoecoico":"hypoechogenicity",
        "isoeocoico":"isoechogenicity",
        "hiperecoico":"hyperechogenicity",
        "very hypoechoic":"marked hypoechogenicity",
        "muy hipoecoico":"marked hypoechogenicity",
        "marked hypoechogenicity":"marked hypoechogenicity",
        "hypoechogenicity":"hypoechogenicity",
        "isoechogenicity":"isoechogenicity",
        "hyperechogenicity":"hyperechogenicity",
    },
    "margins": {
        "well_defined":"well defined",
        "ill_defined":"ill defined",
        "well defined smooth":"well defined smooth",
        "ill- defined":"ill- defined",
        "lisos":"smooth","smooth":"smooth",
        "microlobulated":"microlobulated","macrolobulated":"macrolobulated",
        "irregular":"irregular","spiculated":"spiculated"
    },
    "calcifications": {
        "none":"non","ninguna":"non",
        "microcalcifications":"microcalcifications",
        "macrocalcifications":"macrocalcifications",
        "microcalcification":"microcalcification",
        "macrocalcification":"macrocalcification",
        "peripheral":"rim","peripheral_calcifications":"rim"
    },
    "sex": { "femenino":"F","female":"F","f":"F", "masculino":"M","male":"M","m":"M","u":"u" }
}

def canonicalize_meta(meta: dict, _cat_maps_unused: dict) -> dict:
    """
    Devuelve SOLO strings canónicos para los campos clínicos
    y 'roi'=(x0,y0,x1,y1) si viene en el XML.
    """
    if not isinstance(meta, dict):
        return {}

    # Aliases de clave
    FIELD_ALIASES = {
        "composition": {"composition","composicion","composición","structure","tipo","comp"},
        "echogenicity":{"echogenicity","ecogenicidad","echogenicidad","eco"},
        "margins":     {"margins","margin","margen","margenes","márgenes","bordes"},
        "calcifications":{"calcifications","calcificacion","calcificaciones","calc","microcalcifications","macrocalcifications","peripheral_calcifications"},
        "sex":         {"sex","sexo","gender"},
        "age":         {"age","edad","years","age_years","edad_años"},
        "roi":         {"roi","box","bbox"},
        # esquinas / bbox alternativo
        "xmin":{"xmin"},"ymin":{"ymin"},"xmax":{"xmax"},"ymax":{"ymax"},
        "x":{"x"},"y":{"y"},"w":{"w"},"h":{"h"},
    }

    def _k(raw):
        r = (raw or "").strip().lower()
        for can, pool in FIELD_ALIASES.items():
            if r == can or r in pool:
                return can
        return None

    # llaves y valores como str
    low = {}
    for k,v in meta.items():
        kk = (k or "").strip().lower()
        vv = "" if v is None else str(v).strip()
        low[kk] = vv

    out = {}

    # Clinicos como string
    for rawk, rawv in low.items():
        can = _k(rawk)
        if can in ("composition","echogenicity","margins","calcifications"):
            v = rawv.lower()
            # aplica alias -> string canónico
            v = _VALUE_ALIASES.get(can, {}).get(v, v)
            if v:
                out[can] = v
        elif can == "sex":
            v = rawv.strip().lower()
            out["sex"] = "F" if v.startswith("f") else ("M" if v.startswith("m") else None)
        elif can == "age":
            # devolver string (compatibilidad)
            s = rawv.strip()
            if s:
                # normaliza a entero en string si se puede
                try: out["age"] = str(int(float(s.replace(",", "."))))
                except: out["age"] = s

    # ROI (acepta roi/box/bbox o xmin,ymin,xmax,ymax o x,y,w,h)
    def _num(x):
        try: return float(str(x).strip().replace(",", "."))
        except: return None

    def _tuple4(x):
        if isinstance(x,(list,tuple)) and len(x)==4: return tuple(x)
        if isinstance(x,str):
            s = x.strip()
            if not s or s.lower() in ("none","nan","null"): return None
            try:
                v = ast.literal_eval(s)
                if isinstance(v,(list,tuple)) and len(v)==4:
                    return tuple(v)
            except: return None
        return None

    roi = None
    for k in ("roi","box","bbox"):
        if k in low and low[k]:
            t = _tuple4(low[k])
            if t: roi = t; break

    if roi is None and all(k in low for k in ("xmin","ymin","xmax","ymax")):
        x0=_num(low["xmin"]); y0=_num(low["ymin"])
        x1=_num(low["xmax"]); y1=_num(low["ymax"])
        if None not in (x0,y0,x1,y1): roi = (x0,y0,x1,y1)

    if roi is None and all(k in low for k in ("x","y","w","h")):
        x=_num(low["x"]); y=_num(low["y"]); w=_num(low["w"]); h=_num(low["h"])
        if None not in (x,y,w,h) and w>0 and h>0:
            roi = (x,y,x+w,y+h)

    if roi: out["roi"] = roi
    return out

# Normalizacion estandar de ImageNet (la que usa ResNet-18)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_img_tfms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# Orden fijo de campos clinicos
FIELDS = ["composition", "echogenicity", "margins", "calcifications", "sex"]

def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts

def preprocess_image(img_path: str, xml_path: str | None = None):
    """
    Abre la imagen y devuelve un tensor [3,224,224] ya normalizado.
    El wrapper anhade la dimension batch si hace falta.
    """
    img = Image.open(img_path).convert("RGB")
    x = _img_tfms(img)
    return x

def load_cat_maps(path):
    """Carga el mapping de categorias usado para one-hot. Si no existe, usa un default."""
    if not os.path.exists(path):
        return {
            "composition": {"solid":0, "mixed":1, "cystic":2},
            "echogenicity": {
                "hypoechoic":0, "isoechoic":1, "hyperechogenicity":2,
                "very_hypoechoic":3, "hyperechoic":2
            },
            "margins": {
                "smooth":0, "ill_defined":1, "irregular":2,
                "extrathyroidal":3, "well_defined":4, "well defined":4
            },
            "calcifications": {"none":0, "non":0, "micro":1, "macro":2, "peripheral":3},
            "sex": {"F":0, "M":1}
        }
    return json.load(open(path, "r", encoding="utf-8"))

def compute_tab_dim(cat_maps: dict) -> int:
    """Dimension total del vector tabular = suma de one-hots + 1 (age)."""
    return sum(len(list(cat_maps[k].keys())) for k in FIELDS) + 1

def encode_clinical(payload: dict, cat_maps: dict) -> list[float]:
    """
    Convierte form/XML a vector tabular [one-hots..., age].
    - Si un campo falta/no coincide -> one-hot en ceros.
    - Toleramos numericos como '2' si la key es '2'.
    - Edad: 0.0 si falta o es invalida (neutral, sin medias).
    """
    vec = []
    for k in FIELDS:
        keys = list(cat_maps[k].keys())
        one = [0.0] * len(keys)
        v = str(payload.get(k) or "").strip()
        if v in keys:
            one[keys.index(v)] = 1.0
        elif v != "" and (v in keys or (v.isdigit() and str(int(v)) in keys)):
            key = v if v in keys else str(int(v))
            one[keys.index(key)] = 1.0
        vec.extend(one)

    raw_age = str(payload.get("age") or "").strip()
    try:
        age_val = float(raw_age) if raw_age != "" else 0.0
    except:
        age_val = 0.0
    vec.append(age_val)
    return vec

def encode_clinical_with_mask(payload: dict, cat_maps: dict):
    """
    Devuelve (vec, coverage, k_present)
      - vec: [one-hots..., age] con 0.0 en faltantes
      - coverage ∈ [0,1]: fraccion de grupos presentes (5 categorias + edad)
      - k_present ∈ [0..6]: numero entero de grupos presentes
    """
    vec = []
    groups_present = 0

    for k in FIELDS:
        keys = list(cat_maps[k].keys())
        one = [0.0] * len(keys)
        v = str(payload.get(k) or "").strip()
        if v in keys:
            one[keys.index(v)] = 1.0
            groups_present += 1
        elif v.isdigit() and (str(int(v)) in keys):
            key = str(int(v))
            one[keys.index(key)] = 1.0
            groups_present += 1
        vec.extend(one)

    age_raw = str(payload.get("age") or "").strip()
    age_present = False
    try:
        if age_raw != "":
            age_val = float(age_raw)
            age_present = True
        else:
            age_val = 0.0
    except:
        age_val = 0.0
    vec.append(age_val)

    k_total = len(FIELDS) + 1
    k_present = groups_present + (1 if age_present else 0)
    coverage = k_present / k_total
    return vec, coverage, k_present

def _parse_xml_meta(xml_path: str) -> dict:
    meta = {}
    if not xml_path or not os.path.exists(xml_path):
        return meta
    try:
        root = ET.parse(xml_path).getroot()
        # Recorre TODOS los "descendientes", no solo "hijos" directos
        for elem in root.iter():
            if elem is root:
                continue
            key = (elem.tag or "").split('}')[-1].strip().lower()
            val = (elem.text or "").strip()
            if key and val:
                meta[key] = val
        return meta
    except Exception:
        return {}
    
def recompute_and_update_record(rec: dict, clinical_payload: dict):
    """
    Recalcula usando la MISMA logica de ROI que /predict:
    - XML -> U-Net -> center
    y actualiza roi_source = 'xml' | 'unet' | 'center' segun lo usado.
    """
    wrapper = ctx.wrapper
    if wrapper is None:
        abort(503, "El modelo no está listo o no se pudo cargar.")

    if not rec.get("file"):
        abort(400, "El registro no tiene una imagen asociada para re-calcular.")

    up_dir = current_app.config['UPLOAD_FOLDER']
    image_path = os.path.join(up_dir, rec["file"])
    if not os.path.exists(image_path):
        abort(404, f"No se encontró el archivo de imagen: {rec['file']}")

    # Antes que nada, persistir clinicos que llegaron
    for k in ("composition","echogenicity","margins","calcifications","sex","age"):
        v = clinical_payload.get(k)
        if v not in (None, "", "-"):
            rec[k] = v

    # ROI consistente (XML -> U-Net -> center)
    xml_path = None
    if rec.get("xml") and rec["xml"] != "-":
        cand = os.path.join(up_dir, rec["xml"])
        if os.path.exists(cand):
            xml_path = cand

    # Usa el mismo helper que /predict
    _prepare_image_with_roi = ctx._prepare_image_with_roi
    try:
        x_img, roi_src, box, full_rgb, _ = _prepare_image_with_roi(image_path, xml_path)
    except Exception as e:
        abort(500, f"Error al preparar imagen/ROI: {e}")

    # Vector clinico tolerante a tipos
    vec, coverage, k_present = encode_clinical_with_mask(clinical_payload, ctx.cat_maps)
    clin_t = torch.tensor(vec, dtype=torch.float32) if any(vec) else None
    if clin_t is not None:
        clin_t = clin_t.unsqueeze(0).to(wrapper.device)

    # Prediccion
    with wrapper.inference_mode():
        _, prob = wrapper.predict(x_img, clin_t)

    tau_img = getattr(wrapper, "tau_img", 0.5)
    tau_mm  = getattr(wrapper, "tau_mm", 0.5)
    tau_by_k = getattr(wrapper, "tau_by_k", {})
    tau = float(tau_by_k.get(str(int(k_present)), tau_img + coverage * (tau_mm - tau_img)))

    pred  = 1 if float(prob) >= float(tau) else 0
    label = "Maligno" if pred == 1 else "Benigno"

    rec.update({
        "label": label,
        "pred": int(pred),
        "prob": round(float(prob), 4),
        "meta_used": k_present > 0,
        "meta_coverage": round(float(coverage), 3),
        "k_present": int(k_present),
        "roi_source": (roi_src or "center"),
        "box": box if box else "-",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def _count_present_fields(payload: dict) -> int:
    """Cuenta cuántos campos clínicos están presentes en el payload."""
    if not payload:
        return 0
    
    keys_to_check = {"composition", "echogenicity", "margins", "calcifications", "sex", "age"}
    return sum(1 for k in keys_to_check if payload.get(k))