from PIL import Image
import os
import shutil
import csv

# 1. Resize Images
folder = "C:/Users/Eichleitner/Documents/Coding/TestValidation/"
target_size = (256, 256)
output_folder = "C:/Users/Eichleitner/Documents/Coding/TestValidation/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for root, dirs, files in os.walk(folder):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(root, filename)
            image = Image.open(filepath)
            resized_image = image.resize(target_size)
            rel_path = os.path.relpath(root, folder)
            output_subfolder = os.path.join(output_folder, rel_path)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_path = os.path.join(output_subfolder, filename)
            resized_image.save(output_path)
            print(f"Resized {filepath} to {target_size} and saved to {output_path}")

# 2. Convert Formats
directory = output_folder
extension = '.png'
new_extension = '.jpg'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(extension):
            with Image.open(os.path.join(root, file)) as im:
                new_file = os.path.splitext(file)[0] + new_extension
                new_path = os.path.join(root, new_file)
                im.convert('RGB').save(new_path)
                os.remove(os.path.join(root, file))

# 3. Rename and Categorize
input_folder = output_folder
output_folder_categorized = r"C:/Users/Eichleitner/Documents/Coding/TestValidationCategorized"

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith(".jpg"):
            subfolder_name = os.path.basename(root)
            image_name = os.path.splitext(filename)[0]
            new_filename = f"{subfolder_name}_{image_name}{os.path.splitext(filename)[1]}"
            if not os.path.exists(output_folder_categorized):
                os.makedirs(output_folder_categorized)
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_folder_categorized, new_filename)
            shutil.copy(input_path, output_path)
            print(f"Renamed and copied {input_path} to {output_path}")

# 4. Create Labels CSV
input_folder_categorized = output_folder_categorized
output_csv = r"C:/Users/Eichleitner/Documents/Coding/labelsTestValidation.csv"
file_groups = {}

for filename in os.listdir(input_folder_categorized):
    if filename.endswith(".jpg"):
        parts = filename.split('_')
        label = parts[0]
        if label in file_groups:
            file_groups[label].append(filename)
        else:
            file_groups[label] = [filename]

rows = []

for label, filenames in file_groups.items():
    filenames.sort()
    for i, filename in enumerate(filenames):
        rows.append([i+1, input_folder_categorized, filename, label])

rows.sort(key=lambda x: (x[3], x[0]))

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Filepath', 'Filename', 'Label'])
    writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")
