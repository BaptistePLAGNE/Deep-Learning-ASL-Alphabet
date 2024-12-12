import os
from rembg import remove
from PIL import Image
import io

def remove_background_recursive(root_folder):
    # Parcours de tous les sous-dossiers et fichiers
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Vérifier si le fichier est une image (extensions courantes)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    print(f"Processing: {file_path}")
                    with open(file_path, "rb") as input_file:
                        input_image = input_file.read()
                        output_image = remove(input_image)  # Suppression du fond

                    # Convertir en format PNG
                    image = Image.open(io.BytesIO(output_image))
                    output_path = file_path  # Remplacer l'image d'origine
                    image.save(output_path, format="PNG")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Dossier racine à traiter
    root_folder = "./01_data"

    if not os.path.exists(root_folder):
        print(f"The folder '{root_folder}' does not exist.")
    else:
        remove_background_recursive(root_folder)
        print("Processing complete.")