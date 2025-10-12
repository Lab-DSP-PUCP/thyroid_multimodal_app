# db_models.py
from datetime import date, datetime
from sqlalchemy import CheckConstraint, ForeignKey, event # type: ignore
from sqlalchemy.orm import object_session # type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore

db = SQLAlchemy()


# ----------------------
# Entidad Paciente
# ----------------------
class Patient(db.Model):
    __tablename__ = "patients"

    id          = db.Column(db.Integer, primary_key=True)
    code        = db.Column(db.String(64), unique=True, nullable=False, index=True)
    full_name   = db.Column(db.String(128))
    sex         = db.Column(db.String(8))     # 'F' / 'M' (u otro si lo necesitas)
    birth_date  = db.Column(db.Date)
    phone       = db.Column(db.String(32))
    notes       = db.Column(db.Text)

    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at  = db.Column(db.DateTime)

    __table_args__ = (
        CheckConstraint("length(trim(code)) > 0", name="ck_pat_code_nonempty"),
    )


# ----------------------
# Resultado / Examen
# ----------------------
class Result(db.Model):
    __tablename__ = "results"
    id            = db.Column(db.Integer, primary_key=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at    = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient_id    = db.Column(db.Integer, ForeignKey("patients.id"), nullable=False)

    image_path    = db.Column(db.String(512))
    xml_path      = db.Column(db.String(512))
    roi_path      = db.Column(db.String(512))
    roi_source    = db.Column(db.String(16))    # 'xml' | 'unet' | 'meta' | 'manual'

    pred_label    = db.Column(db.String(16))
    pred_prob     = db.Column(db.Float)
    used_metadata = db.Column(db.Boolean, default=False)

    # metadatos clínicos que ALIMENTAN al modelo
    sex           = db.Column(db.String(1))     # 'F'/'M'
    age           = db.Column(db.Integer)
    composition   = db.Column(db.String(64))
    echogenicity  = db.Column(db.String(64))
    margins       = db.Column(db.String(64))
    calcifications= db.Column(db.String(64))

    # otros
    exam_notes    = db.Column(db.Text)
    ti_rads       = db.Column(db.Integer)
    manual_label  = db.Column(db.String(16))

    deleted_at    = db.Column(db.DateTime)

    __table_args__ = (
        CheckConstraint("pred_prob IS NULL OR (pred_prob>=0 AND pred_prob<=1)", name="ck_prob_0_1"),
        CheckConstraint("age IS NULL OR (age>=0 AND age<=120)", name="ck_age_range"),
    )

    def apply_clinical_updates(self, **kw) -> bool:
        """
        Aplica cambios en campos que afectan la inferencia.
        Devuelve True si cambió al menos uno (=> recompute).
        """
        mapping = {
            "sex": "sex",
            "age": "age",
            "composition": "composition",
            "echogenicity": "echogenicity",
            "margins": "margins",
            "calcifications": "calcifications",
            "notes": "exam_notes",
        }
        changed = False
        for k_form, attr in mapping.items():
            if k_form in kw:
                val = kw.get(k_form)
                if isinstance(val, str) and val.strip() == "":
                    val = None
                if getattr(self, attr) != val:
                    setattr(self, attr, val)
                    changed = True
        return changed


# ----------------------
# Normalización ligera (sin autocalcular)
# ----------------------
def _clamp_age(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v <= 120 else None

@event.listens_for(Result, "before_insert")
def result_before_insert(mapper, connection, target: Result):
    # Sexo a 'F'/'M' si corresponde
    if isinstance(target.sex, str):
        s = target.sex.strip().upper()[:1]
        target.sex = s if s in ("F", "M") else None
    # Edad válida si llegó
    target.age = _clamp_age(target.age)

@event.listens_for(Result, "before_update")
def result_before_update(mapper, connection, target: Result):
    if isinstance(target.sex, str):
        s = target.sex.strip().upper()[:1]
        target.sex = s if s in ("F", "M") else None
    target.age = _clamp_age(target.age)