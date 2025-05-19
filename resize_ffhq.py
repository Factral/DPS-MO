from PIL import Image
import os
import glob

def resize_ffhq_recursive(source_root_dir, target_root_dir, target_size=(256, 256)):
    """
    Redimensiona imágenes PNG del dataset FFHQ recursivamente desde source_root_dir
    y las guarda en target_root_dir manteniendo la estructura de subdirectorios.

    Args:
        source_root_dir (str): El directorio raíz que contiene las imágenes originales
                               (ej: "ffhq-dataset/images1024x1024").
        target_root_dir (str): El directorio raíz donde se guardarán las imágenes
                               redimensionadas (ej: "ffhq-dataset/images256x256").
        target_size (tuple): El tamaño deseado para las imágenes (ancho, alto).
    """

    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
        print(f"Directorio de destino creado: {target_root_dir}")

    # Usamos glob con '**/*.png' y recursive=True para encontrar todas las imágenes png
    # en todos los subdirectorios.
    # os.path.join es importante para la compatibilidad entre sistemas operativos.
    search_pattern = os.path.join(source_root_dir, "**", "*.png")
    image_paths = glob.glob(search_pattern, recursive=True)

    if not image_paths:
        print(f"No se encontraron imágenes PNG en: {search_pattern}")
        return

    print(f"Se encontraron {len(image_paths)} imágenes para procesar.")

    for img_path in image_paths:
        try:
            # Obtener la ruta relativa de la imagen con respecto al directorio fuente
            # Esto nos ayudará a replicar la estructura en el directorio destino
            relative_path = os.path.relpath(img_path, source_root_dir)
            
            # Construir la ruta de destino completa
            target_img_path = os.path.join(target_root_dir, relative_path)
            
            # Crear el subdirectorio de destino si no existe
            target_img_subdir = os.path.dirname(target_img_path)
            if not os.path.exists(target_img_subdir):
                os.makedirs(target_img_subdir)

            # Abrir, redimensionar y guardar la imagen
            img = Image.open(img_path)
            # Asegurarse de que la imagen sea RGB para evitar problemas con PNGs con canal alfa o escala de grises
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            resized_img = img.resize(target_size, Image.LANCZOS) # O Image.BICUBIC
            resized_img.save(target_img_path)
            # print(f"Redimensionada y guardada: {target_img_path}")

        except Exception as e:
            print(f"No se pudo procesar {img_path}: {e}")

    print("Proceso de redimensionamiento completado.")

# --- Configuración ---
# Cambia estas rutas según tu estructura de archivos
SOURCE_IMAGES_DIR = "/ibex/user/perezpnf/ffhq-dataset/images1024x1024"  # Donde están tus imágenes 1024x1024
TARGET_IMAGES_DIR = "/ibex/user/perezpnf/ffhq-dataset/images256x256"    # Donde quieres guardar las imágenes 256x256
TARGET_RESOLUTION = (256, 256)

# --- Ejecutar el script ---
if __name__ == "__main__":
    # Verifica si el directorio fuente existe
    if not os.path.isdir(SOURCE_IMAGES_DIR):
        print(f"Error: El directorio fuente '{SOURCE_IMAGES_DIR}' no existe.")
        print("Asegúrate de que la ruta sea correcta y que hayas descargado el dataset FFHQ.")
    else:
        resize_ffhq_recursive(SOURCE_IMAGES_DIR, TARGET_IMAGES_DIR, TARGET_RESOLUTION)
        print(f"\nLas imágenes redimensionadas deberían estar en: {os.path.abspath(TARGET_IMAGES_DIR)}")
        print("Puedes usar esta ruta como 'root' en tu configuración de datos (data.root) en los archivos YAML.")