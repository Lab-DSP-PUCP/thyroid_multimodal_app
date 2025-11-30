## 🚀 Uso rápido (Windows) — sin instalar nada
1) Descarga **ThyroidAid.exe** desde [Releases](https://github.com/Lab-DSP-PUCP/thyroid_multimodal_app/releases). 
2) Doble click y abre **http://127.0.0.1:5000**.

> El ejecutable incluye el modelo y funciona **offline**.  
> Carpeta de imágenes y meta-datos de prueba: [DDTI](https://drive.google.com/drive/folders/17OMDOmK8qCGn3IjPJb1XcFY56NasNGZY?usp=drive_link).
> Carpeta de imágenes y meta-datos de prueba: [TN5000](https://drive.google.com/drive/folders/1llhw7Q5Hhuzzzjp6-eDUpN5eBvd6V6v_).

---

## 👩‍💻 Modo desarrollador (opcional)
```bat
git clone https://github.com/Lab-DSP-PUCP/thyroid_multimodal_app.git
cd thyroid_multimodal_app
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Si se quiere generar el ejecutable
pyinstaller --noconfirm --clean --onefile --name ThyroidAid --add-data "templates;templates" --add-data "static;static" --add-data "models;models" app.py 
A veces, si no genera, es mejor primero limpiar:
```bat
taskkill /IM ThyroidAid.exe /F 2>$null
>> taskkill /IM python.exe /F 2>$null
>> Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
>> Remove-Item -Recurse -Force .\dist  -ErrorAction SilentlyContinue
```
