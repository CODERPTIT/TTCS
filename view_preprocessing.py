import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

path = r'd:\review_ttcs_12th5\BraTS2024_UNet_SegFormer.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'def resize_volume' in src or 'def crop_brain_roi' in src or 'def build_dataloaders' in src:
            print(f"Cell index: {idx}")
            print("="*40)
            print(src)
            print("="*40)
