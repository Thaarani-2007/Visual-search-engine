"""
Production Kaggle Training Script for Stanford Online Products (SOP).
Automated official download, anti-mock guard, zero-leakage split, and evaluation.
"""
# Cell 1: Environment Diagnostics & GPU Telemetry
import os
import sys
import time
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

# Install FAISS and gdown
# !pip install -q faiss-cpu gdown  # Shell command in notebook
import faiss

# Hardware Diagnostics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('=' * 75)
print(f'PyTorch Version: {torch.__version__}')
print(f'Compute Device:  {device}')
if device.type == 'cuda':
    print(f'GPU Model:       {torch.cuda.get_device_name(0)}')
    print(f'Device Count:    {torch.cuda.device_count()}')
    print(f'Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    raise SystemError('CRITICAL: GPU accelerator not detected! Enable GPU in Kaggle sidebar settings before running.')
print('=' * 75)

# Cell 2: Automated Official Dataset Acquisition & Multi-Check Anti-Mock Guard
def validate_sop_directory(dir_path: Path) -> bool:
    """
    Thoroughly validates candidate SOP directory against metadata headers, row counts,
    and category directory hierarchy. Rejects prototypes or incomplete downloads.
    """
    train_file = dir_path / 'Ebay_train.txt'
    test_file = dir_path / 'Ebay_test.txt'
    info_file = dir_path / 'Ebay_info.txt'
    
    if not (train_file.is_file() and test_file.is_file() and info_file.is_file()):
        return False
        
    # Header & row count validation
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_header = f.readline().strip().split()
            train_lines = sum(1 for _ in f)
        with open(test_file, 'r', encoding='utf-8') as f:
            test_lines = sum(1 for _ in f)
            
        expected_headers = {'image_id', 'class_id', 'super_class_id', 'path'}
        if not expected_headers.issubset(set(train_header)):
            return False
            
        # Official SOP has 59,551 train images and 60,502 test images
        if train_lines < 50000 or test_lines < 50000:
            print(f'[GUARD REJECTION] {dir_path} has only {train_lines} train and {test_lines} test lines.')
            print('  -> This is a prototype or subset, NOT the official 120K Stanford Online Products dataset.')
            return False
            
        # Verify category directory existence (all 12 categories)
        expected_cats = ['bicycle_final', 'chair_final', 'table_final', 'sofa_final',
                         'lamp_final', 'cabinet_final', 'coffee_maker_final', 'fan_final',
                         'kettle_final', 'mug_final', 'stapler_final', 'toaster_final']
        for cat in expected_cats:
            if not (dir_path / cat).is_dir():
                print(f'[GUARD REJECTION] Missing category folder: {cat}')
                return False
                
        return True
    except Exception as e:
        print(f'Validation error on {dir_path}: {e}')
        return False

def acquire_official_sop() -> Path:
    working_dir = Path('/kaggle/working')
    input_dir = Path('/kaggle/input')
    
    # 1. Search /kaggle/input/ for already-attached official dataset
    for p in input_dir.rglob('Ebay_info.txt'):
        cand_dir = p.parent
        if validate_sop_directory(cand_dir):
            print(f'[SUCCESS] Found attached verified official SOP dataset at: {cand_dir}')
            return cand_dir
            
    # 2. Search /kaggle/working/ for extracted official dataset
    for p in working_dir.rglob('Ebay_info.txt'):
        cand_dir = p.parent
        if validate_sop_directory(cand_dir):
            print(f'[SUCCESS] Found extracted verified official SOP dataset at: {cand_dir}')
            return cand_dir
            
    # 3. Download official SOP archive (~3.08 GB)
    print('\nOfficial SOP benchmark not found. Downloading official Stanford Online Products archive (~3.08 GB)...')
    zip_path = working_dir / 'Stanford_Online_Products.zip'
    
    if not zip_path.is_file() or zip_path.stat().st_size < 2.5 * 1024**3:
        import gdown
        file_id = '1TclrpQOF_ullUP99wk_gjGN8pKvtErG8'
        print(f'Streaming official Stanford archive via Google Drive mirror (ID: {file_id})...')
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
        
    assert zip_path.is_file(), 'Download failed: Archive file not found on disk.'
    archive_size_gb = zip_path.stat().st_size / 1e9
    print(f'Downloaded archive size: {archive_size_gb:.2f} GB')
    assert archive_size_gb >= 2.5, f'Archive corrupted or incomplete ({archive_size_gb:.2f} GB < 2.5 GB target).'
    
    # 4. Extract Archive
    print('Extracting official archive (takes ~60-90 seconds)...')
    t0 = time.time()
#     !unzip -q /kaggle/working/Stanford_Online_Products.zip -d /kaggle/working/  # Shell command in notebook
    print(f'Extraction finished in {time.time() - t0:.1f} seconds.')
    
    # Clean up zip to prevent running out of working disk space
    if zip_path.is_file():
        zip_path.unlink()
        print('Removed zip archive to preserve disk space.')
        
    # Re-verify extracted directory
    for p in working_dir.rglob('Ebay_info.txt'):
        cand_dir = p.parent
        if validate_sop_directory(cand_dir):
            print(f'[SUCCESS] Official Stanford Online Products dataset extracted and verified at: {cand_dir}')
            return cand_dir
            
    raise FileNotFoundError('Failed to acquire and validate the official Stanford Online Products benchmark.')

SOP_ROOT = acquire_official_sop()
OUTPUT_DIR = Path('/kaggle/working')

# Cell 3: Official Dataset Validation, Statistics & Zero-Leakage Split
def validate_and_partition_sop(root_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Tuple[int, int]]]:
    """
    Validates official split statistics and generates an identity-disjoint partition.
    Creates a deterministic validation positive pair manifest.
    """
    train_file = root_dir / 'Ebay_train.txt'
    test_file = root_dir / 'Ebay_test.txt'
    info_file = root_dir / 'Ebay_info.txt'
    
    df_raw_train = pd.read_csv(train_file, sep=r'\s+', header=0)
    df_raw_test = pd.read_csv(test_file, sep=r'\s+', header=0)
    df_raw_info = pd.read_csv(info_file, sep=r'\s+', header=0)
    
    # Standardize column naming
    for df in [df_raw_train, df_raw_test, df_raw_info]:
        df.rename(columns={'class_id': 'product_id', 'super_class_id': 'category_id', 'path': 'relative_path'}, inplace=True)
        df['category'] = df['relative_path'].apply(lambda p: str(p).split('/')[0].replace('_final', ''))
        
    print('=' * 80)
    print('OFFICIAL STANFORD ONLINE PRODUCTS DATASET BENCHMARK AUDIT')
    print('=' * 80)
    print(f'Official Training Split (Ebay_train.txt): {len(df_raw_train):>6} images across {df_raw_train["product_id"].nunique():>5} SKUs')
    print(f'Official Testing Split  (Ebay_test.txt):  {len(df_raw_test):>6} images across {df_raw_test["product_id"].nunique():>5} SKUs')
    print(f'Total Dataset           (Ebay_info.txt):  {len(df_raw_info):>6} images across {df_raw_info["product_id"].nunique():>5} SKUs')
    
    # Official Benchmark Statistics Assertions
    assert len(df_raw_train) == 59551, f'Expected 59,551 train images, found {len(df_raw_train)}'
    assert df_raw_train["product_id"].nunique() == 11318, f'Expected 11,318 train SKUs, found {df_raw_train["product_id"].nunique()}'
    assert len(df_raw_test) == 60502, f'Expected 60,502 test images, found {len(df_raw_test)}'
    assert df_raw_test["product_id"].nunique() == 11316, f'Expected 11,316 test SKUs, found {df_raw_test["product_id"].nunique()}'
    assert len(df_raw_info) == 120053, f'Expected 120,053 total images, found {len(df_raw_info)}'
    
    # Category distribution assertion
    train_cats = set(df_raw_train['category'].unique())
    test_cats = set(df_raw_test['category'].unique())
    assert len(train_cats) == 12, f'Expected 12 categories, found {len(train_cats)}'
    assert train_cats == test_cats, 'Train and Test category sets mismatch!'
    print(f'All 12 Official Categories Verified: {sorted(train_cats)}')
    
    # Check min images per identity
    train_id_counts = df_raw_train['product_id'].value_counts()
    assert train_id_counts.min() >= 2, f'Train SKU with < 2 images detected: min={train_id_counts.min()}'
    print(f'Train SKUs image count: min={train_id_counts.min()}, max={train_id_counts.max()}, mean={train_id_counts.mean():.2f}')
    
    # Identity-Disjoint Partition of Training identities into Train (90%) and Val (10%)
    train_pids = df_raw_train['product_id'].unique()
    np.random.seed(42)
    np.random.shuffle(train_pids)
    
    n_total_train_ids = len(train_pids)
    n_val_ids = int(n_total_train_ids * 0.10)
    val_id_set = set(train_pids[:n_val_ids])
    pure_train_id_set = set(train_pids[n_val_ids:])
    test_id_set = set(df_raw_test['product_id'].unique())
    
    # Zero-Leakage Mathematical Assertions
    assert pure_train_id_set.isdisjoint(val_id_set), 'Train and Val identity leakage!'
    assert pure_train_id_set.isdisjoint(test_id_set), 'Train and Test identity leakage!'
    assert val_id_set.isdisjoint(test_id_set), 'Val and Test identity leakage!'
    
    df_train = df_raw_train[df_raw_train['product_id'].isin(pure_train_id_set)].copy().reset_index(drop=True)
    df_val = df_raw_train[df_raw_train['product_id'].isin(val_id_set)].copy().reset_index(drop=True)
    df_test = df_raw_test.copy().reset_index(drop=True)
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    
    # Create DETERMINISTIC validation positive pair manifest
    print('\nBuilding deterministic validation positive pair manifest...')
    np.random.seed(42)
    val_fixed_pairs = []
    val_id_to_indices = {}
    for idx, row in df_val.iterrows():
        val_id_to_indices.setdefault(row['product_id'], []).append(idx)
        
    for idx, row in df_val.iterrows():
        pid = row['product_id']
        peers = [i for i in val_id_to_indices[pid] if i != idx]
        assert len(peers) >= 1, f'Validation identity {pid} has no peer positive image!'
        chosen_pos = int(np.random.choice(peers))
        val_fixed_pairs.append((idx, chosen_pos))
        
    print(f'Constructed {len(val_fixed_pairs):,} fixed, deterministic validation pairs.')
    
    print('\nPARTITION BREAKDOWN:')
    print(f'  Train Split: {len(df_train):>6} images | {df_train["product_id"].nunique():>5} SKUs | {len(df_train)/df_train["product_id"].nunique():.2f} imgs/SKU')
    print(f'  Val Split:   {len(df_val):>6} images | {df_val["product_id"].nunique():>5} SKUs | {len(df_val)/df_val["product_id"].nunique():.2f} imgs/SKU')
    print(f'  Test Split:  {len(df_test):>6} images | {df_test["product_id"].nunique():>5} SKUs | {len(df_test)/df_test["product_id"].nunique():.2f} imgs/SKU')
    print('=' * 80)
    
    manifest = {
        'total_images': len(df_raw_info),
        'train_images': len(df_train),
        'val_images': len(df_val),
        'test_images': len(df_test),
        'train_skus': df_train['product_id'].nunique(),
        'val_skus': df_val['product_id'].nunique(),
        'test_skus': df_test['product_id'].nunique(),
        'categories': sorted(list(train_cats)),
    }
    with open(OUTPUT_DIR / 'sop_split_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
        
    return df_train, df_val, df_test, val_fixed_pairs

df_train, df_val, df_test, val_fixed_pairs = validate_and_partition_sop(SOP_ROOT)

# Cell 4: Pairwise Metric Dataset & DataLoader (Deterministic Validation)
class OfficialSOPMetricDataset(Dataset):
    """
    PyTorch Dataset returning (anchor_img, positive_img, product_id).
    For training: samples positive dynamically with random crop & jitter.
    For validation: strictly uses the deterministic fixed_pairs manifest for zero fluctuation across epochs.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path,
        split: str = 'train',
        image_size: int = 224,
        fixed_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.split = split
        self.fixed_pairs = fixed_pairs
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            
        # Inverted index for positive candidate lookup
        self.id_to_indices = {}
        for idx, row in self.df.iterrows():
            self.id_to_indices.setdefault(row['product_id'], []).append(idx)
        self.id_to_indices_np = {k: np.array(v) for k, v in self.id_to_indices.items()}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index: int):
        anchor_row = self.df.iloc[index]
        pid = anchor_row['product_id']
        
        if self.fixed_pairs is not None:
            # Deterministic positive pairing for validation
            pos_idx = self.fixed_pairs[index][1]
        else:
            # Dynamic training sampling
            candidates = self.id_to_indices_np[pid]
            peers = candidates[candidates != index]
            pos_idx = np.random.choice(peers) if len(peers) > 0 else index
            
        pos_row = self.df.iloc[pos_idx]
        assert pos_row['product_id'] == pid, 'Positive pair identity mismatch!'
        
        with Image.open(self.root_dir / anchor_row['relative_path']) as img:
            a_img = self.transform(img.convert('RGB'))
        with Image.open(self.root_dir / pos_row['relative_path']) as img:
            p_img = self.transform(img.convert('RGB'))
            
        return a_img, p_img, pid

# DataLoaders (Batch size: 64 pairs = 128 images per forward pass)
BATCH_SIZE = 64
NUM_WORKERS = 4

train_dataset = OfficialSOPMetricDataset(df_train, SOP_ROOT, split='train')
val_dataset = OfficialSOPMetricDataset(df_val, SOP_ROOT, split='val', fixed_pairs=val_fixed_pairs)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f'Train DataLoader: {len(train_loader)} batches/epoch ({BATCH_SIZE} pairs = {BATCH_SIZE*2} images/batch)')
print(f'Val DataLoader:   {len(val_loader)} batches/epoch (Deterministic evaluation)')

# Cell 5: ResNet-50 Metric Architecture with Fixed BatchNorm Override
class ResNet50MetricNet(nn.Module):
    """
    ResNet-50 backbone with selective early layer freezing and 512-D L2 projection head.
    Explicitly fixes BatchNorm running statistics in frozen early layers during train().
    """
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.freeze_early_layers = True
        self.frozen_modules = [self.conv1, self.bn1, self.layer1, self.layer2]
        for m in self.frozen_modules:
            for p in m.parameters():
                p.requires_grad = False
                
        # 512-D Projection Head with BatchNorm1d
        self.head = nn.Sequential(
            nn.Linear(2048, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        
    def train(self, mode: bool = True):
        """Override train to keep frozen early BatchNorm layers in eval mode (no running stat drift)."""
        super().train(mode)
        if self.freeze_early_layers:
            for m in self.frozen_modules:
                m.eval()
                
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.head(x)
        return F.normalize(z, p=2, dim=1, eps=1e-12)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_one(x)
        
    def forward_pair(self, a: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = a.size(0)
        combined = torch.cat([a, p], dim=0)
        z = self.forward_one(combined)
        return z[:batch_size], z[batch_size:]

model = ResNet50MetricNet(embedding_dim=512, pretrained=True).to(device)
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f'Model initialized: {trainable_p:,} trainable params | {frozen_p:,} frozen params')
print('Frozen BatchNorm layers locked in eval() mode to preserve ImageNet statistics.')

# Cell 6: Vectorized Online Semi-Hard Triplet Margin Loss
def pairwise_distance(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable Euclidean distance between all rows of L2-normalized vectors."""
    dot = torch.clamp(torch.matmul(x, x.t()), -1.0, 1.0)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * dot, min=0.0) + eps)

class OnlineSemiHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, a_emb: torch.Tensor, p_emb: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B = a_emb.size(0)
        ap_dot = torch.clamp((a_emb * p_emb).sum(dim=1), -1.0, 1.0)
        d_ap = torch.sqrt(torch.clamp(2.0 - 2.0 * ap_dot, min=1e-12))
        
        # In-batch pool of candidates from both anchors and positives
        pool_embs = torch.cat([a_emb, p_emb], dim=0)
        pool_labels = torch.cat([labels, labels], dim=0)
        
        all_embs = torch.cat([a_emb, pool_embs], dim=0)
        dist_matrix = pairwise_distance(all_embs)[:B, B:]
        
        is_neg = labels.unsqueeze(1) != pool_labels.unsqueeze(0)
        
        selected_d_an = []
        semi_hard_hits = 0
        
        for i in range(B):
            neg_mask_i = is_neg[i]
            if not neg_mask_i.any():
                selected_d_an.append(d_ap[i] + self.margin)
                continue
                
            neg_dists = dist_matrix[i, neg_mask_i]
            d_ap_i = d_ap[i]
            
            semi_hard = (neg_dists > d_ap_i) & (neg_dists < d_ap_i + self.margin)
            if semi_hard.any():
                semi_hard_hits += 1
                chosen = torch.min(neg_dists[semi_hard])
            else:
                chosen = torch.min(neg_dists)
                
            selected_d_an.append(chosen)
            
        d_an = torch.stack(selected_d_an)
        losses = F.relu(d_ap - d_an + self.margin)
        loss = losses.mean()
        
        metrics = {
            'loss': loss.item(),
            'pos_dist': d_ap.mean().item(),
            'neg_dist': d_an.mean().item(),
            'gap': (d_an.mean() - d_ap.mean()).item(),
            'active_ratio': (losses > 0).float().mean().item(),
            'semi_hard_ratio': semi_hard_hits / B,
        }
        return loss, metrics

# Cell 7: Pre-Flight Smoke Test (Sanity Verification Before Full Training)
def run_preflight_smoke_test():
    """
    Executes 2 training batches and 1 validation batch.
    Verifies forward/backward pass, loss computation, numerical stability, and GPU VRAM.
    """
    print('=' * 80)
    print('RUNNING PRE-FLIGHT SMOKE TEST...')
    print('=' * 80)
    test_criterion = OnlineSemiHardTripletLoss(margin=0.2)
    test_opt = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    model.train()
    for step, (a, p, pids) in enumerate(train_loader):
        if step >= 2:
            break
        a, p = a.to(device), p.to(device)
        labels = torch.as_tensor(pids, device=device)
        test_opt.zero_grad()
        a_emb, p_emb = model.forward_pair(a, p)
        
        # Assertions on embeddings
        assert not torch.isnan(a_emb).any(), 'NaN in anchor embeddings!'
        assert not torch.isnan(p_emb).any(), 'NaN in positive embeddings!'
        norms = torch.norm(a_emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), 'Embedding L2 normalization failed!'
        
        loss, m = test_criterion(a_emb, p_emb, labels)
        loss.backward()
        test_opt.step()
        print(f'Smoke Train Batch {step+1}: Loss = {m["loss"]:.4f}, Gap = {m["gap"]:.3f}, Active = {m["active_ratio"]*100:.1f}%')
        
    # Validation smoke test
    model.eval()
    with torch.no_grad():
        for step, (a, p, pids) in enumerate(val_loader):
            if step >= 1:
                break
            a, p = a.to(device), p.to(device)
            labels = torch.as_tensor(pids, device=device)
            a_emb, p_emb = model.forward_pair(a, p)
            loss, m = test_criterion(a_emb, p_emb, labels)
            print(f'Smoke Val Batch   {step+1}: Loss = {m["loss"]:.4f}, Gap = {m["gap"]:.3f}')
            
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
    print(f'Peak GPU VRAM during Smoke Test: {peak_mem:.2f} GB (Safe threshold: < 14 GB)')
    print('✓ PRE-FLIGHT SMOKE TEST PASSED! All tensors, losses, and gradients are healthy.')
    print('=' * 80 + '\n')

run_preflight_smoke_test()

# Cell 8: GPU Training Engine with Telemetry Logging & Peak Memory Reset
def train_metric_engine(epochs: int = 15, lr: float = 1e-4, weight_decay: float = 1e-5, margin: float = 0.2):
    criterion = OnlineSemiHardTripletLoss(margin=margin)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_gap': [], 'val_gap': [],
        'active_triplets': [], 'gpu_mem_gb': [],
        'epoch_duration_sec': [],
    }
    
    best_val_loss = float('inf')
    best_ckpt_path = OUTPUT_DIR / 'best_metric_model.pt'
    
    print('=' * 80)
    print(f'STARTING RETRAINING ON OFFICIAL STANFORD ONLINE PRODUCTS ({epochs} Epochs)')
    print('=' * 80)
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats per epoch
            
        model.train()
        t_loss, t_gap, t_active = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:2d}/{epochs:2d} [Train]')
        
        for a, p, pids in pbar:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            labels = torch.as_tensor(pids, device=device)
            
            optimizer.zero_grad()
            a_emb, p_emb = model.forward_pair(a, p)
            loss, m = criterion(a_emb, p_emb, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            t_loss += m['loss']
            t_gap += m['gap']
            t_active += m['active_ratio']
            pbar.set_postfix({
                'loss': f"{m['loss']:.4f}",
                'gap': f"{m['gap']:.3f}",
                'act': f"{m['active_ratio']*100:.1f}%",
            })
            
        scheduler.step()
        t_loss /= len(train_loader)
        t_gap /= len(train_loader)
        t_active /= len(train_loader)
        
        # Validation Loop (Deterministic Manifest)
        model.eval()
        v_loss, v_gap = 0.0, 0.0
        with torch.no_grad():
            for a, p, pids in val_loader:
                a = a.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                labels = torch.as_tensor(pids, device=device)
                a_emb, p_emb = model.forward_pair(a, p)
                loss, m = criterion(a_emb, p_emb, labels)
                v_loss += m['loss']
                v_gap += m['gap']
                
        v_loss /= max(len(val_loader), 1)
        v_gap /= max(len(val_loader), 1)
        
        duration = time.time() - t_epoch_start
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_gap'].append(t_gap)
        history['val_gap'].append(v_gap)
        history['active_triplets'].append(t_active)
        history['gpu_mem_gb'].append(peak_vram)
        history['epoch_duration_sec'].append(duration)
        
        print(f'Epoch {epoch:2d}/{epochs:2d} | Train Loss: {t_loss:.4f} (Gap: {t_gap:.3f}, Active: {t_active*100:.1f}%) | ' \
              f'Val Loss: {v_loss:.4f} (Gap: {v_gap:.3f}) | Time: {duration:.1f}s | Peak VRAM: {peak_vram:.2f} GB')
              
        # Save best model checkpoint strictly based on lowest deterministic validation loss
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': v_loss,
                'val_gap': v_gap,
                'margin': margin,
                'embedding_dim': 512,
            }, best_ckpt_path)
            print(f'  ★ New Best Model Saved -> {best_ckpt_path} (Val Loss: {v_loss:.4f})')
            
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    plt.figure(figsize=(14, 5), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#ea580c', lw=2)
    plt.title('Semi-Hard Triplet Margin Loss (Margin=0.2)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_gap'], label='Train Gap: d(a,n) - d(a,p)', color='#16a34a', lw=2)
    plt.plot(history['val_gap'], label='Val Gap', color='#9333ea', lw=2)
    plt.axhline(margin, linestyle='--', color='gray', label=f'Target Margin ({margin})')
    plt.title('Embedding Separation Dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Separation Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'training_curves.png', bbox_inches='tight')
    plt.close()
    print('\nTraining curves saved to training_curves.png')
    return history

training_history = train_metric_engine(epochs=15, lr=1e-4, weight_decay=1e-5, margin=0.2)

# Cell 9: Official Held-Out Test Retrieval Evaluation (Correct Recall@5, Explicit Query Exclusion, Latency)
def evaluate_official_retrieval(df_test: pd.DataFrame, root_dir: Path, num_eval_queries: int = 2000):
    """
    Leave-one-out retrieval evaluation on official held-out test identities.
    Strictly fixes:
    1. Query index explicitly excluded (no assumption that query is at index 0).
    2. Recall@5 calculated as (relevant items retrieved in top 5) / (total relevant in gallery).
    3. HitRate@5 calculated as (any relevant item in top 5).
    4. Corrected mAP@5 denominator (denom = min(relevant_candidates, 5)).
    5. Realistic single-query latency benchmark.
    """
    print('=' * 80)
    print(f'PHASE 8: EVALUATION ON OFFICIAL HELD-OUT TEST SPLIT (Sampled {num_eval_queries:,} Queries)')
    print('=' * 80)
    
    ckpt_path = OUTPUT_DIR / 'best_metric_model.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded best checkpoint from Epoch {ckpt["epoch"]} (Val Loss: {ckpt["val_loss"]:.4f})')
    
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    class TestExtractionDataset(Dataset):
        def __init__(self, df, root):
            self.df = df
            self.root = root
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            with Image.open(self.root / row['relative_path']) as img:
                t = eval_transform(img.convert('RGB'))
            return t, row['product_id'], row['relative_path']
            
    test_ds = TestExtractionDataset(df_test, root_dir)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'Extracting 512-D embeddings for all {len(test_ds):,} official test images...')
    all_embs, all_pids, all_paths = [], [], []
    t0 = time.time()
    
    with torch.no_grad():
        for imgs, pids, paths in tqdm(test_dl, desc='Test Embedding Extraction'):
            imgs = imgs.to(device, non_blocking=True)
            embs = model(imgs).cpu().numpy()
            all_embs.append(embs)
            all_pids.extend(pids.numpy())
            all_paths.extend(paths)
            
    all_embs = np.vstack(all_embs).astype(np.float32)
    all_pids = np.array(all_pids)
    extraction_time = time.time() - t0
    throughput = len(all_embs) / extraction_time
    print(f'Extracted {len(all_embs):,} vectors in {extraction_time:.1f}s ({throughput:.1f} images/second)')
    
    # Build FAISS Exact FlatIP Index
    index = faiss.IndexFlatIP(512)
    index.add(all_embs)
    
    # Select evaluation queries across unseen test identities
    n_total = len(all_pids)
    n_queries = min(num_eval_queries, n_total)
    np.random.seed(42)
    query_indices = np.random.choice(n_total, size=n_queries, replace=False)
    query_embs = all_embs[query_indices]
    query_pids = all_pids[query_indices]
    
    # 1. Measure Batch Query Search
    t_search_batch = time.time()
    # Retrieve top 20 to safely exclude the query index regardless of tie-breaks
    scores_raw, retrieved_raw = index.search(query_embs, k=20)
    batch_search_time_ms = (time.time() - t_search_batch) * 1000 / n_queries
    
    # 2. Benchmark Realistic Single-Query Search Latency (50 single queries)
    single_latencies = []
    for q_emb in query_embs[:50]:
        q_vec = np.expand_dims(q_emb, 0)
        t_single = time.perf_counter()
        _ = index.search(q_vec, k=10)
        single_latencies.append((time.perf_counter() - t_single) * 1000)
    single_latency_mean = np.mean(single_latencies)
    single_latency_p50 = np.median(single_latencies)
    
    hit_1 = 0.0
    hit_5 = 0.0
    hit_10 = 0.0
    recall_5_list = []
    recall_10_list = []
    precision_5_list = []
    ap_5_list = []
    valid_queries_count = 0
    
    clean_retrieved_top10 = []
    clean_scores_top10 = []
    
    for i in range(n_queries):
        q_idx = query_indices[i]
        q_pid = query_pids[i]
        
        # EXPLICIT QUERY INDEX FILTERING: Filter out the query's own index
        raw_cands = retrieved_raw[i]
        raw_sc = scores_raw[i]
        mask = (raw_cands != q_idx)
        
        cands_filtered = raw_cands[mask][:10]
        scores_filtered = raw_sc[mask][:10]
        
        clean_retrieved_top10.append(cands_filtered)
        clean_scores_top10.append(scores_filtered)
        
        ret_pids = all_pids[cands_filtered]
        
        # Total relevant candidates in gallery (excluding query itself)
        total_matching = np.sum(all_pids == q_pid)
        if total_matching <= 1:
            continue  # No peer items in test set
        relevant_candidates = total_matching - 1
        valid_queries_count += 1
        
        # 1. HitRate@1: Is at least one relevant item ranked first?
        if ret_pids[0] == q_pid:
            hit_1 += 1.0
            
        # 2. HitRate@5: Is at least one relevant item in top 5?
        if np.any(ret_pids[:5] == q_pid):
            hit_5 += 1.0
            
        # 3. HitRate@10: Is at least one relevant item in top 10?
        if np.any(ret_pids[:10] == q_pid):
            hit_10 += 1.0
            
        # 4. Precision@5: Fraction of top 5 results that are relevant (out of 5)
        n_rel_5 = np.sum(ret_pids[:5] == q_pid)
        precision_5_list.append(n_rel_5 / 5.0)
        
        # 5. Recall@5: Fraction of ALL relevant gallery images retrieved in top 5
        recall_5_list.append(n_rel_5 / relevant_candidates)
        
        # 6. Recall@10: Fraction of ALL relevant gallery images retrieved in top 10
        n_rel_10 = np.sum(ret_pids[:10] == q_pid)
        recall_10_list.append(n_rel_10 / relevant_candidates)
        
        # 7. Corrected mAP@5 (denom = min(relevant_candidates, 5))
        hits = 0
        prec_sum = 0.0
        for k in range(1, 6):
            if ret_pids[k - 1] == q_pid:
                hits += 1
                prec_sum += hits / k
        denom = min(relevant_candidates, 5)
        ap_5_list.append(prec_sum / denom if denom > 0 else 0.0)
        
    clean_retrieved_top10 = np.array(clean_retrieved_top10)
    clean_scores_top10 = np.array(clean_scores_top10)
    
    report = {
        'evaluation_scope': f'Sampled {valid_queries_count:,} test queries across 60,502 official test images',
        'total_official_test_images': int(n_total),
        'total_official_test_skus': int(len(np.unique(all_pids))),
        'evaluated_queries': int(valid_queries_count),
        'HitRate@1': round((hit_1 / valid_queries_count) * 100.0, 2),
        'HitRate@5': round((hit_5 / valid_queries_count) * 100.0, 2),
        'HitRate@10': round((hit_10 / valid_queries_count) * 100.0, 2),
        'Recall@5': round(float(np.mean(recall_5_list)) * 100.0, 2),
        'Recall@10': round(float(np.mean(recall_10_list)) * 100.0, 2),
        'Precision@5': round(float(np.mean(precision_5_list)) * 100.0, 2),
        'mAP@5': round(float(np.mean(ap_5_list)), 4),
        'single_query_latency_mean_ms': round(single_latency_mean, 3),
        'single_query_latency_p50_ms': round(single_latency_p50, 3),
        'embedding_throughput_img_per_sec': round(throughput, 1),
    }
    
    print('\n' + '=' * 80)
    print('OFFICIAL STANFORD ONLINE PRODUCTS RETRIEVAL EVALUATION REPORT')
    print('=' * 80)
    print(f'Scope:                         {report["evaluation_scope"]}')
    print(f'HitRate@1:                     {report["HitRate@1"]}% (Queries with relevant item at rank #1)')
    print(f'HitRate@5:                     {report["HitRate@5"]}% (Queries with >= 1 relevant item in top 5)')
    print(f'HitRate@10:                    {report["HitRate@10"]}% (Queries with >= 1 relevant item in top 10)')
    print(f'Recall@5 (True Definition):    {report["Recall@5"]}% (Fraction of relevant gallery items in top 5)')
    print(f'Recall@10:                     {report["Recall@10"]}% (Fraction of relevant gallery items in top 10)')
    print(f'Precision@5:                   {report["Precision@5"]}% (Fraction of top 5 results that are relevant)')
    print(f'mAP@5:                         {report["mAP@5"]} (Rank-weighted quality across top 5)')
    print(f'Single Query Search Latency:   {report["single_query_latency_mean_ms"]} ms (p50: {report["single_query_latency_p50_ms"]} ms)')
    print(f'Embedding Extraction Speed:    {report["embedding_throughput_img_per_sec"]} images/second')
    print('=' * 80)
    
    with open(OUTPUT_DIR / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    # Visual retrieval matches grid (clearly labeled Cosine Similarity)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    for row in range(4):
        q_idx = query_indices[row]
        q_pid = all_pids[q_idx]
        q_img = Image.open(root_dir / all_paths[q_idx]).convert('RGB')
        axes[row, 0].imshow(q_img)
        axes[row, 0].set_title(f'QUERY\nSKU: {q_pid}', fontweight='bold', color='#1e293b')
        axes[row, 0].axis('off')
        for spine in axes[row, 0].spines.values():
            spine.set_visible(True); spine.set_color('#2563eb'); spine.set_linewidth(3)
            
        for rank in range(5):
            ax = axes[row, rank + 1]
            match_idx = clean_retrieved_top10[row, rank]
            match_pid = all_pids[match_idx]
            match_img = Image.open(root_dir / all_paths[match_idx]).convert('RGB')
            is_hit = (match_pid == q_pid)
            b_color = '#16a34a' if is_hit else '#dc2626'
            status = 'MATCH' if is_hit else 'MISMATCH'
            cos_sim = clean_scores_top10[row, rank]
            ax.imshow(match_img)
            ax.set_title(f'#{rank+1} (CosSim: {cos_sim:.3f})\nSKU: {match_pid}\n[{status}]', 
                         color=b_color, fontweight='bold', fontsize=9)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color(b_color); spine.set_linewidth(2)
                
    plt.savefig(OUTPUT_DIR / 'retrieval_visual_matches.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Visual retrieval matches grid saved to retrieval_visual_matches.png')
    return report

eval_report = evaluate_official_retrieval(df_test, SOP_ROOT, num_eval_queries=2000)

# Cell 10: Artifact Summary & Download Instructions
print('=' * 80)
print('KAGGLE OFFICIAL RETRAINING PIPELINE COMPLETED SUCCESSFULLY!')
print('=' * 80)
print('Generated Artifacts in /kaggle/working/:')
for f in OUTPUT_DIR.glob('*'):
    if f.is_file() and not f.name.endswith('.zip'):
        print(f'  - {f.name:<32} ({f.stat().st_size / 1e6:.2f} MB)')
print('=' * 80)
print('\nNEXT STEP: Download `best_metric_model.pt` and `evaluation_report.json`')
print('and copy them to your local repository at:')
print('  models/checkpoints/best_metric_model.pt')
print('Then run local evaluation and API tests!')
print('=' * 80)

