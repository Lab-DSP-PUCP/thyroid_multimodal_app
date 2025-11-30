# Verificador rapido de dependencias
import importlib, sys

mods = [
    "flask", "sqlalchemy", "flask_sqlalchemy", "dotenv",
    "PIL", "numpy", "cv2", "matplotlib",
    "torch", "torchvision"
]

missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, repr(e)))

if missing:
    print("[ERROR] Faltan módulos:")
    for m, err in missing:
        print(f"  - {m}: {err}")
    sys.exit(1)

print("[OK] Todas las dependencias cargaron correctamente.")
