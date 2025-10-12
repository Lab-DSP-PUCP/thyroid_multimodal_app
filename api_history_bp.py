import os, json, datetime, torch # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from flask import Blueprint, request, jsonify, abort, current_app, session  # type: ignore
from core import ctx, canonicalize_meta, _parse_xml_meta, recompute_and_update_record
from utils import allowed_file  # si lo necesitas solo a esto, ok

bp = Blueprint("history_api", __name__, url_prefix="/api/history")

def _history_path():
    up = current_app.config.get("UPLOAD_FOLDER")
    if not up:
        abort(500, description="UPLOAD_FOLDER no está configurado.")
    return os.path.join(up, "_history.json")

def _load_history():
    p = _history_path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(h):
    p = _history_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

def _require_admin_session():
    if not session.get("is_admin"):
        abort(401, description="Requiere iniciar sesión de administrador.")


ALLOWED_UPDATE_FIELDS = {
    "patient_id", "composition", "echogenicity", "margins",
    "calcifications", "sex", "age"
}

PROHIBITED_FIELDS = {"ti_rads", "manual_label", "label", "pred", "prob"}

# RUTA DE ACTUALIZACION (METODO POST)
@bp.route("/<uid>", methods=["POST"])
def update_item(uid):
    _require_admin_session()
    history = _load_history()
    target_item = next((item for item in history if str(item.get("id")) == str(uid)), None)
    if not target_item:
        abort(404, description="No se encontró el elemento.")

    xml_file = request.files.get('xmlfile')
    clinical_data = {}

    # --- A) Si suben XML nuevo, se guarda y se recalcula ---
    if xml_file and xml_file.filename != '':
        filename = secure_filename(f"{uid}_meta_{int(datetime.datetime.now().timestamp())}.xml")
        xml_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        xml_file.save(xml_path)
        target_item['xml'] = filename

        raw_meta = _parse_xml_meta(xml_path)
        clinical_data = canonicalize_meta(raw_meta, ctx.cat_maps)

    # --- B) Si NO suben XML, se usan datos manuales ---
    else:
        clinical_data = {
            "composition": request.form.get("composition"),
            "echogenicity": request.form.get("echogenicity"),
            "margins": request.form.get("margins"),
            "calcifications": request.form.get("calcifications"),
            "sex": request.form.get("sex"),
            "age": request.form.get("age"),
        }
    
    # Estado actual y acción del request
    has_xml_now   = bool(target_item.get('xml') and target_item['xml'] != '-')
    uploading_xml = bool(xml_file and xml_file.filename)
    remove_xml    = request.form.get("remove_xml") in ("1", "true", "on", "yes")

    # ¿llegó clínica manual?
    provided_data = {k: v for k, v in clinical_data.items() if v}

    # Si tiene XML y no lo están quitando ni reemplazando, NO aceptar manuales
    if has_xml_now and not uploading_xml and not remove_xml and provided_data:
        return jsonify({
            "error": "Este registro ya tiene XML.",
            "detail": "Para editar manualmente, marca 'Quitar XML actual' o sube un nuevo XML."
        }), 400

    # === C) (PUNTO 7) Permitir quitar el XML explícitamente ===
    if remove_xml:
        target_item["xml"] = "-"
        # opcional: también limpiar ROI derivado del XML
        # target_item["roi_preview"] = "-"
        # target_item["roi_source"] = None

    # Recalcular si llegó algo clínico (manual o de XML)
    provided_data = {k: v for k, v in clinical_data.items() if v}
    if provided_data:
        try:
            recompute_and_update_record(target_item, provided_data)
        except Exception as e:
            return jsonify({"error": f"Falló el recálculo: {e}"}), 400

    # Campos no clínicos
    target_item['patient_id'] = request.form.get('patient_id', target_item.get('patient_id'))

    _save_history(history)
    return jsonify(target_item)

def _try_remove(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass

@bp.route("/<uid>/recompute", methods=["POST"])
def recompute_history(uid):
    _require_admin_session()
    # Funciones/objetos del app
    _prepare_image_with_roi = ctx._prepare_image_with_roi
    encode_clinical_with_mask = ctx.encode_clinical_with_mask
    wrapper = ctx.wrapper
    cat_maps = ctx.cat_maps

    hist = _load_history()
    rec = next((h for h in hist if str(h.get("id")) == str(uid)), None)
    if not rec:
        return {"error": "Registro no encontrado"}, 404

    # 0) ID paciente obligatorio
    pid = request.form.get("patient_id", "").strip()
    if not pid:
        return {"error": "El ID de paciente es obligatorio."}, 400
    rec["patient_id"] = pid

    # 1) Lee manuales (solo no vacíos serán overrides)
    manual = {k: (request.form.get(k, "").strip() or None)
              for k in ("composition","echogenicity","margins","calcifications","sex","age")}

    # 2) Manejo de XML
    new_xml = request.files.get("xmlfile")
    remove_xml = request.form.get("remove_xml") in ("1","true","on","yes")
    xml_path = None

    if new_xml and new_xml.filename and allowed_file(new_xml.filename, {"xml"}):
        # a) Suben XML nuevo → cargar todo desde XML, luego manuales pisan
        saved_xml = f"{uid}_{new_xml.filename.replace(' ', '_')}"
        xml_path = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_xml)
        new_xml.save(xml_path)
        rec["xml"] = saved_xml

        raw = _parse_xml_meta(xml_path)
        meta = canonicalize_meta(raw, cat_maps)
        for k in ("composition","echogenicity","margins","calcifications","sex","age"):
            v = meta.get(k)
            if v not in (None, "", "-"):
                rec[k] = v

        for k, v in manual.items():
            if v not in (None, "", "-"):
                rec[k] = v

    elif remove_xml:
        # b) Quitar XML ⇒ limpiar todos los clínicos a '-' y luego manuales pisan
        rec["xml"] = "-"
        for k in ("composition","echogenicity","margins","calcifications","sex","age"):
            rec[k] = "-"                      # usa "" si prefieres vacío real
        for k, v in manual.items():
            if v not in (None, "", "-"):
                rec[k] = v

    else:
        # c) Mantener XML previo (si hay). Solo manuales no vacíos pisan.
        for k, v in manual.items():
            if v not in (None, "", "-"):
                rec[k] = v
        if rec.get("xml") and rec["xml"] != "-":
            xml_path = os.path.join(current_app.config['UPLOAD_FOLDER'], rec["xml"])

    # 3) ROI / imagen
    img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], rec["file"])
    try:
        x_img, roi_src, box, full_rgb, _ = _prepare_image_with_roi(img_path, xml_path)
    except Exception as e:
        return {"error": f"Error al procesar la imagen/ROI: {e}"}, 500
    if xml_path and box:
        roi_src = "xml"   # etiqueta clara si se usó box de XML

    # 4) Clínica → vector → predicción
    clin_payload = {k: rec.get(k, "") for k in ("composition","echogenicity","margins","calcifications","sex","age")}
    # Normaliza: '-' y '' => None (para que no cuente como presente)
    for k, v in list(clin_payload.items()):
        if v in (None, "", "-"):
            clin_payload[k] = None
    
    vec, coverage, k_present = encode_clinical_with_mask(clin_payload, cat_maps)
    clin_t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(wrapper.device) if any(vec) else None

    with wrapper.inference_mode():
        _, prob = wrapper.predict(x_img, clin_t)

    tau_img = getattr(wrapper, "tau_img", 0.5)
    tau_mm  = getattr(wrapper, "tau_mm", 0.5)
    tau_by_k = getattr(wrapper, "tau_by_k", {})
    tau = float(tau_by_k.get(str(int(k_present)), tau_img + coverage * (tau_mm - tau_img)))
    pred  = 1 if float(prob) >= float(tau) else 0

    rec.update({
        "label": "Maligno" if pred == 1 else "Benigno",
        "pred": int(pred),
        "prob": round(float(prob), 4),
        "meta_used": k_present > 0,
        "roi_source": roi_src,
        "box": box if box else "-",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    _save_history(hist)
    return jsonify(rec)

@bp.delete("/<uid>")
def delete_item(uid):
    _require_admin_session()
    purge = request.args.get("purge") == "1"
    history = _load_history()
    up = current_app.config.get("UPLOAD_FOLDER")
    found = False

    for i, item in enumerate(history):
        if str(item.get("id")) == str(uid):
            found = True
            if purge:
                # Borrar archivos fisicos asociados
                file = item.get("file")
                xml  = item.get("xml")
                roi  = item.get("roi_preview")
                if file and file != "-":
                    _try_remove(os.path.join(up, file))
                if xml and xml != "-":
                    _try_remove(os.path.join(up, xml))
                if roi and roi != "-":
                    _try_remove(os.path.join(up, roi))
                # Quitar del historial
                history.pop(i)
            else:
                item["deleted_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            _save_history(history)
            return jsonify({"ok": True})
    if not found:
        abort(404, description="No se encontró el elemento.")