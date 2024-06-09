import os
import shutil

# Set source and destination paths
pdsf_path = '\\\\vihsdv004.eu.infineon.com\\eqdatvi\\THIN\\SPINETCH_SEZ_RST_TRC'
equipment = 'RST5C'
VIEWERPATH = 'VIEWERPATH'
file_path = os.path.join(pdsf_path, equipment, VIEWERPATH)
destination = 'C:\\Users\\Eichleitner\\Documents\\Coding\\pdsf'

print("source: ", file_path)

# Exclude these files
excluded_files = ['PREFLUSH.pdsf', 'CDS1.pdsf', 'CDS2.pdsf', 'CDS3.pdsf']

# Get all .pdsf files in the source directory
files = [f for f in os.listdir(file_path) if f.endswith('.pdsf') and f not in excluded_files]

# Sort files by modified time (newest first)
files.sort(key=lambda f: os.path.getmtime(os.path.join(file_path, f)), reverse=True)

# Copy the 5 newest files
for file in files[:5]:
    shutil.copy(os.path.join(file_path, file), destination)
    print(f'Copied {file} to {destination}')
