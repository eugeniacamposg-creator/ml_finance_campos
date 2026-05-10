import os

# Lista de carpetas que exige el profesor
directorios = [
    "data/raw", 
    "data/processed", 
    "notebooks", 
    "src", 
    "models/baseline_v1", 
    "reports/figures"
]

for d in directorios:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, ".gitkeep"), "w") as f:
        pass

# Archivo de inicialización para que Python reconozca la carpeta src
with open("src/__init__.py", "w") as f:
    pass

print("¡CARPETAS CREADAS CON ÉXITO EN EL ESCRITORIO!")