import os
from PIL import Image

def resize_images_in_folder(root_folder, size=(96, 96)):
    """ Recursively resizes all images in the given root folder and its subfolders. """
    
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            
            file_path = os.path.join(subdir, file)
            
            try:
                # Open image
                with Image.open(file_path) as img:
                    # Convert to RGB (to handle all formats properly)
                    img = img.convert("RGB")
                    # Resize
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)
                    # Save (overwrite the original image)
                    img_resized.save(file_path)
                    print(f"Resized: {file_path}")
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

# Example Usage
root_directory = "Test/"  # Change this to your folder path
resize_images_in_folder(root_directory)
