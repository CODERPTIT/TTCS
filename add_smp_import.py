import json

path = r'd:\review_ttcs_12th5\BraTS2024_UNet_SegFormer.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first code cell is nb['cells'][1]
cell = nb['cells'][1]
source = cell['source']

# Find if smp is already in source
smp_imported = False
for line in source:
    if 'segmentation_models_pytorch' in line:
        smp_imported = True
        break

if not smp_imported:
    # Insert import at the beginning
    source.insert(0, "import segmentation_models_pytorch as smp\n")
    print("Added smp import to Cell 1")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook updated successfully!")
else:
    print("smp is already imported in Cell 1.")
