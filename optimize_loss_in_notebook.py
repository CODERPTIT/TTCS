import json

path = r'd:\review_ttcs_12th5\BraTS2024_UNet_SegFormer.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update CONFIG in Cell 2
updated_config_code = """CONFIG = {
    # Data
    'data_root': '/kaggle/input/datasets/i212385nomanarif/2024-brats-glioma',
    'modalities': ['t1n', 't1c', 't2w', 't2f'],
    'seg_suffix': 'seg',
    
    # 2.5D settings
    'k_2p5d': 2,          # neighbor slices each side
    'n_slices': 5,        # 2*k + 1
    'in_channels': 20,    # 4 modalities x 5 slices
    
    # Image
    'slice_size': 192,
    
    # Classes: 0=BG, 1=NETC, 2=SNFH, 3=ET, 4=RC
    'num_classes': 5,
    'class_names': ['Background', 'NETC', 'SNFH', 'ET', 'RC'],
    
    # Model
    'base_channels': 32,
    
    # Training
    'lr': 1e-4,
    'weight_decay': 1e-4,  # [TỐI ƯU] Tăng chống Overfitting cho Transformer
    'batch_size': 8,      # adjust to GPU
    'num_epochs': 200,
    
    # Split
    'train_split': 0.70,
    'val_split': 0.10,
    'test_split': 0.20,
    'seed': 99,
    
    # Loss weights [TỐI ƯU MỚI: Tăng Tversky Loss để giải quyết phân đoạn u nhỏ]
    'loss_focal_tversky_w': 0.6,
    'loss_ce_w': 0.2,
    'loss_boundary_w': 0.2,
    
    # Saving
    'checkpoint_dir': './checkpoints',
    'best_model_path': './checkpoints/best_unet2p5d.pth',
}

import os, random
import numpy as np
import torch
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
print('Config loaded.')
print(f"In channels: {CONFIG['in_channels']}  |  Out classes: {CONFIG['num_classes']}")"""

# 2. Update Loss definition in Cell 5
updated_loss_code = """class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, ignore_bg=True):
        super().__init__()
        self.C = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.C).permute(0, 3, 1, 2).float()
        
        start_c = 1 if self.ignore_bg else 0
        dice_per_class = []
        for c in range(start_c, self.C):
            p = probs[:, c]
            g = one_hot[:, c]
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            dice_per_class.append(1.0 - dice)
        return torch.stack(dice_per_class).mean()

class FocalTverskyLoss(nn.Module):
    \"\"\"
    Advanced Loss for highly imbalanced medical data.
    Tối ưu hóa: alpha=0.2, beta=0.8 (Phạt rất nặng lỗi bỏ sót False Negatives của u nhỏ)
    \"\"\"
    def __init__(self, num_classes, smooth=1e-5, ignore_bg=True, alpha=0.2, beta=0.8, gamma=0.75):
        super().__init__()
        self.C = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg
        self.alpha = alpha  # FP penalty weight
        self.beta = beta    # FN penalty weight (tập trung tránh bỏ sót u)
        self.gamma = gamma  # Focal parameter
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.C).permute(0, 3, 1, 2).float()
        
        start_c = 1 if self.ignore_bg else 0
        tversky_per_class = []
        
        for c in range(start_c, self.C):
            p = probs[:, c]
            g = one_hot[:, c]
            
            TP = (p * g).sum()
            FP = (p * (1 - g)).sum()
            FN = ((1 - p) * g).sum()
            
            tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            focal_tversky = (1 - tversky_index) ** self.gamma
            tversky_per_class.append(focal_tversky)
            
        return torch.stack(tversky_per_class).mean()

class FastBoundaryLoss(nn.Module):
    \"\"\"
    A fast, GPU-native boundary loss using Sobel Edge Detection.
    \"\"\"
    def __init__(self, num_classes):
        super().__init__()
        self.C = num_classes
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        
    def get_edges(self, x):
        edge_x = F.conv2d(x, self.kx, padding=1)
        edge_y = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.C).permute(0, 3, 1, 2).float()
        
        b_loss = 0.0
        for c in range(1, self.C):
            p = probs[:, c:c+1]
            g = one_hot[:, c:c+1]
            edge_p = self.get_edges(p)
            edge_g = self.get_edges(g)
            b_loss += F.mse_loss(edge_p, edge_g)
            
        return b_loss / (self.C - 1)

class CombinedAdvancedLoss(nn.Module):
    def __init__(self, num_classes, ft_w=0.6, ce_w=0.2, bd_w=0.2):
        super().__init__()
        # Cấu hình tham số tối ưu hóa trực tiếp
        self.focal_tversky = FocalTverskyLoss(num_classes, alpha=0.2, beta=0.8, gamma=0.75)
        self.ce = nn.CrossEntropyLoss()
        self.boundary = FastBoundaryLoss(num_classes)
        
        self.ft_w = ft_w
        self.ce_w = ce_w
        self.bd_w = bd_w
    
    def forward(self, logits, targets):
        ft = self.focal_tversky(logits, targets)
        c = self.ce(logits, targets)
        bd = self.boundary(logits, targets)
        
        total = self.ft_w * ft + self.ce_w * c + self.bd_w * bd
        return total, ft.item(), c.item()

print('Optimized Advanced Loss functions defined.')"""

# Thay thế code
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'CONFIG = {' in src:
            cell['source'] = [line + '\n' for line in updated_config_code.split('\n')]
            if cell['source']: cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"Updated CONFIG in Cell {idx}")
        elif 'class CombinedAdvancedLoss' in src:
            cell['source'] = [line + '\n' for line in updated_loss_code.split('\n')]
            if cell['source']: cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"Updated Loss functions in Cell {idx}")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully with optimized Loss parameters!")
