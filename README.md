## 🚀 Uso rápido (Windows) — sin instalar nada
1) Descarga **ThyroidAid.exe** desde [Releases](../releases/latest).
2) Doble click y abre **http://127.0.0.1:5000**.

> El ejecutable incluye el modelo y funciona **offline**.  
> (Opcional) Carpeta de imágenes de prueba: [Drive](<tu_enlace_de_drive>).

---

## 👩‍💻 Modo desarrollador (opcional)
```bat
git clone https://github.com/Lab-DSP-PUCP/thyroid_multimodal_app.git
cd thyroid_multimodal_app
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py