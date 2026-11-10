"""
Multi-GPU DDP Training Pipeline (ResNet50 + Discriminative LR + Low CPU Default)
รองรับ: Single GPU / Multi-GPU (DistributedDataParallel)
ฟีเจอร์: Class Weights, Mixup/CutMix (ปิดช่วงท้าย), CuPy Zero-copy, Low CPU Usage
Usage:
Single GPU:  python Multi-Cuda-ResNet50.py --data_path Dataset
2 GPUs:      torchrun --nproc_per_node=2 Multi-Cuda-ResNet50.py --data_path Dataset
"""
# ============================================================================
# 📦 SECTION 1: IMPORTS & ENVIRONMENT
# ============================================================================
import os
import sys
import warnings
import time
import argparse
import logging
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import cupy as cp

# --- ✨ Thread Control: ลด CPU usage ให้ต่ำสุด ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
            self.desc = kwargs.get('desc', '')
        def __iter__(self):
            for item in self.iterable:
                yield item
        def set_postfix(self, *a, **kw):
            pass

warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# 🌐 SECTION 2: DISTRIBUTED SETUP / CLEANUP
# ============================================================================
def setup_distributed() -> Tuple[int, int, int, bool]:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank       = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

# ============================================================================
# 📝 SECTION 3: LOGGING
# ============================================================================
def setup_logger(rank: int) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    log_file = f"logs/cuda_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("MangoTrainer")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    if rank == 0:
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

# ============================================================================
# ⚡ SECTION 4: CUDA VERIFICATION
# ============================================================================
def verify_cuda(local_rank: int) -> Tuple[bool, str, torch.device]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device(f'cuda:{local_rank}')
        props  = torch.cuda.get_device_properties(local_rank)
        info = (
            f"CUDA {torch.version.cuda} | "
            f"GPU {local_rank}: {props.name} | "
            f"Compute {props.major}.{props.minor} | "
            f"VRAM {props.total_memory / 1024**3:.2f} GB"
        )
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        info   = "CUDA NOT AVAILABLE → CPU fallback"
    return cuda_available, info, device

# ============================================================================
# 📊 SECTION 5: DATA PIPELINE (✨ LOW CPU DEFAULT)
# ============================================================================
class DataPipeline:
    def __init__(self, img_size: int = 224, num_workers: int = 2, aug_level: str = 'medium'):
        self.img_size    = img_size
        self.num_workers = num_workers
        self.aug_level   = aug_level

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD  = [0.229, 0.224, 0.225]

        if aug_level == 'strong':
            self.train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.RandAugment(num_ops=2, magnitude=7),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        elif aug_level == 'medium':
            self.train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def load_datasets(self, data_path: str, batch_size: int, is_distributed: bool, world_size: int, rank: int):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")

        has_split = any((data_path / d).exists() for d in ['train', 'val', 'test'])
        if has_split:
            train_ds, val_ds, test_ds, class_names = self._load_split(data_path)
        else:
            train_ds, val_ds, test_ds, class_names = self._load_single(data_path)

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
        val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        test_sampler  = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None

        # ✨ LOW CPU DEFAULT: workers=2, prefetch=2, persistent_workers=False
        nw = self.num_workers
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
                                  num_workers=nw, pin_memory=True, prefetch_factor=2 if nw > 0 else None,
                                  persistent_workers=False, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, sampler=val_sampler,
                                  num_workers=nw, pin_memory=True, prefetch_factor=2 if nw > 0 else None,
                                  persistent_workers=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, sampler=test_sampler,
                                  num_workers=nw, pin_memory=True, prefetch_factor=2 if nw > 0 else None,
                                  persistent_workers=False)

        return train_loader, val_loader, test_loader, class_names, train_sampler, val_sampler, test_sampler

    def _load_split(self, root: Path):
        train_dir, val_dir, test_dir = root / 'train', root / 'val', root / 'test'
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing: {train_dir}")
        train_ds = ImageFolder(str(train_dir), transform=self.train_transform)
        class_names = train_ds.classes
        if val_dir.exists():
            val_ds = ImageFolder(str(val_dir), transform=self.val_test_transform)
        else:
            n = len(train_ds)
            ts, vs = int(0.8 * n), n - int(0.8 * n)
            g = torch.Generator(device='cpu').manual_seed(42)
            train_ds, val_ds = random_split(train_ds, [ts, vs], generator=g)
            val_ds.dataset.transform = self.val_test_transform
        td = test_dir if test_dir.exists() else (val_dir if val_dir.exists() else train_dir)
        test_ds = ImageFolder(str(td), transform=self.val_test_transform)
        return train_ds, val_ds, test_ds, class_names

    def _load_single(self, root: Path):
        full = ImageFolder(str(root), transform=self.train_transform)
        class_names = full.classes
        n = len(full)
        ts, vs = int(0.70 * n), int(0.15 * n)
        te = n - ts - vs
        g = torch.Generator(device='cpu').manual_seed(42)
        train_ds, val_ds, test_ds = random_split(full, [ts, vs, te], generator=g)
        val_ds.dataset.transform = self.val_test_transform
        test_ds.dataset.transform = self.val_test_transform
        return train_ds, val_ds, test_ds, class_names

# ============================================================================
# 🧠 SECTION 6: MODEL ARCHITECTURE (RESNET50)
# ============================================================================
def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ============================================================================
# 🎯 SECTION 6.5: DISCRIMINATIVE LEARNING RATES
# ============================================================================
def get_parameter_groups(model: nn.Module, base_lr: float = 0.01, lr_decay: float = 0.1) -> List[Dict]:
    param_groups = []
    
    # Layer 1: conv1 + bn1 (LR ต่ำสุด)
    conv1_params = list(model.conv1.parameters())
    bn1_params = list(model.bn1.parameters())
    param_groups.append({'params': conv1_params + bn1_params, 'lr': base_lr * (lr_decay ** 3), 'name': 'conv1+bn1', 'weight_decay': 1e-4})
    
    # Layer 2: layer1
    param_groups.append({'params': list(model.layer1.parameters()), 'lr': base_lr * (lr_decay ** 2), 'name': 'layer1', 'weight_decay': 1e-4})
    
    # Layer 3: layer2
    param_groups.append({'params': list(model.layer2.parameters()), 'lr': base_lr * (lr_decay ** 1), 'name': 'layer2', 'weight_decay': 1e-4})
    
    # Layer 4: layer3
    param_groups.append({'params': list(model.layer3.parameters()), 'lr': base_lr * (lr_decay ** 0.5), 'name': 'layer3', 'weight_decay': 1e-4})
    
    # Layer 5: layer4
    param_groups.append({'params': list(model.layer4.parameters()), 'lr': base_lr * (lr_decay ** 0.2), 'name': 'layer4', 'weight_decay': 1e-4})
    
    # Layer 6: fc (classifier) - LR สูงสุด
    param_groups.append({'params': list(model.fc.parameters()), 'lr': base_lr, 'name': 'fc', 'weight_decay': 1e-4})
    
    return param_groups

# ============================================================================
# 🛑 SECTION 7: EARLY STOPPING
# ============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience, self.min_delta, self.counter = patience, min_delta, 0
        self.best_loss, self.early_stop, self.best_state = None, False, None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            raw = model.module if isinstance(model, DDP) else model
            self.best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False

# ============================================================================
# 💾 SECTION 8: CHECKPOINT SAVING
# ============================================================================
def save_checkpoint(model: nn.Module, class_names: list, img_size: int, epoch: int, val_loss: float, val_acc: float, train_loss: float, train_acc: float, timestamp: str, model_name: str, ckpt_dir: str = "checkpoints") -> Tuple[str, str]:
    Path(ckpt_dir).mkdir(exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    data = {
        'model_state_dict': {k: v.cpu() for k, v in raw.state_dict().items()},
        'class_names': class_names, 'class_indices': {n: i for i, n in enumerate(class_names)},
        'img_size': img_size, 'num_classes': len(class_names), 'epoch': epoch,
        'val_loss': val_loss, 'val_acc': val_acc, 'train_loss': train_loss, 'train_acc': train_acc,
        'timestamp': timestamp, 'model_architecture': model_name,
        'pytorch_version': torch.__version__, 'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    }
    unique = f"{ckpt_dir}/best_model_epoch_{epoch:03d}_vl_{val_loss:.4f}_va_{val_acc:.4f}.pt"
    latest = f"{ckpt_dir}/best_model.pt"
    torch.save(data, unique); torch.save(data, latest)
    return unique, latest

# ============================================================================
# 🔧 SECTION 9: UTILITY FUNCTIONS
# ============================================================================
def fmt_time(s: float) -> str:
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

def gpu_mem(dev: torch.device) -> str:
    if dev.type == 'cuda':
        a = torch.cuda.memory_allocated(dev.index) / 1024**3
        t = torch.cuda.get_device_properties(dev.index).total_memory / 1024**3
        return f"{a:.2f}/{t:.1f}GB"
    return "CPU"

def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1: return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / world_size

def gather_predictions(preds: torch.Tensor, labels: torch.Tensor, world_size: int, device: torch.device):
    if world_size <= 1:
        return cp.from_dlpack(preds.detach()), cp.from_dlpack(labels.detach())
    local_size = torch.tensor([len(preds)], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    def _pad_gather(t: torch.Tensor):
        padded = torch.zeros(max_size, dtype=t.dtype, device=device)
        padded[:len(t)] = t
        gathered = [torch.zeros(max_size, dtype=t.dtype, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        parts = [cp.from_dlpack(g[:all_sizes[i].item()]) for i, g in enumerate(gathered)]
        return cp.concatenate(parts)
    return _pad_gather(preds), _pad_gather(labels)

def gather_probs(probs_arr: torch.Tensor, world_size: int, device: torch.device) -> cp.ndarray:
    if world_size <= 1:
        return cp.from_dlpack(probs_arr.detach())
    local_size = torch.tensor([probs_arr.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)
    nc = probs_arr.shape[1]
    padded = torch.zeros(max_size, nc, dtype=probs_arr.dtype, device=device)
    padded[:probs_arr.shape[0]] = probs_arr
    gathered = [torch.zeros(max_size, nc, dtype=probs_arr.dtype, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    parts = [cp.from_dlpack(g[:all_sizes[i].item()]) for i, g in enumerate(gathered)]
    return cp.concatenate(parts)

# ============================================================================
# 🎨 SECTION 9.5: MIXUP & CUTMIX
# ============================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# 🚀 SECTION 10: TRAINING LOOP
# ============================================================================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, timestamp, class_names, img_size, rank, world_size, is_distributed, train_sampler, logger,
          use_mixup=True, use_cutmix=True, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5,
          disable_aug_epochs=20, metrics_every_n=5):
    is_main = is_main_process(rank)
    early_stop = EarlyStopping(patience=20)
    scaler = GradScaler() if device.type == 'cuda' else None
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}

    start, best_epoch, best_vl, best_va = time.time(), 0, float('inf'), 0.0
    use_tqdm = TQDM_AVAILABLE and sys.stdout.isatty() and is_main

    for epoch in range(epochs):
        ep_start = time.time()
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        elapsed = time.time() - start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0

        disable_aug = epoch >= (epochs - disable_aug_epochs)
        current_use_mixup = use_mixup and not disable_aug
        current_use_cutmix = use_cutmix and not disable_aug
        compute_full_metrics = (epoch % metrics_every_n == 0) or (epoch == epochs - 1)

        if is_main:
            aug_info = []
            if current_use_mixup: aug_info.append(f"Mixup(α={mixup_alpha})")
            if current_use_cutmix: aug_info.append(f"CutMix(α={cutmix_alpha}, p={cutmix_prob})")
            if disable_aug: aug_info.append("⚠️ DISABLED")
            metrics_tag = "" if compute_full_metrics else f" [light metrics]"
            lr_info = " | ".join([f"{g['name']}={g['lr']:.2e}" for g in optimizer.param_groups])
            logger.info(f"\n┌─ Epoch {epoch+1:3d}/{epochs} | GPU: {gpu_mem(device)} | Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta)}{metrics_tag}")
            logger.info(f"│  🎯 LRs: {lr_info}")

        # ── TRAIN ──
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="  Train", leave=False, ncols=100) if use_tqdm else enumerate(train_loader)

        for bi, (imgs, lbls) in iterator:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            apply_cutmix = current_use_cutmix and (np.random.rand() < cutmix_prob)
            if current_use_mixup and not apply_cutmix:
                imgs, targets_a, targets_b, lam = mixup_data(imgs, lbls, mixup_alpha)
                use_aug = True
            elif apply_cutmix:
                imgs, targets_a, targets_b, lam = cutmix_data(imgs, lbls, cutmix_alpha)
                use_aug = True
            else:
                targets_a, targets_b, lam = lbls, lbls, 1.0
                use_aug = False

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = mixup_criterion(criterion, out, targets_a, targets_b, lam) if use_aug else criterion(out, lbls)

            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            bs = imgs.size(0)
            run_loss += loss.item() * bs

            with torch.no_grad():
                _, pred = out.max(1)
                if use_aug:
                    correct += (lam * (pred == targets_a).sum().item() + (1 - lam) * (pred == targets_b).sum().item())
                else:
                    correct += (pred == lbls).sum().item()
            total += bs

            if compute_full_metrics:
                all_preds.append(pred.detach())
                all_labels.append(lbls.detach())

            if use_tqdm:
                iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.1%}")

        t_loss = reduce_tensor(torch.tensor(run_loss / total, device=device), world_size).item()
        t_acc = reduce_tensor(torch.tensor(correct / total, device=device), world_size).item()

        if compute_full_metrics and all_preds:
            all_preds_t = torch.cat(all_preds)
            all_labels_t = torch.cat(all_labels)
            g_preds, g_labels = gather_predictions(all_preds_t, all_labels_t, world_size, device)
            t_prec, t_rec, _, _ = precision_recall_fscore_support(g_labels.get(), g_preds.get(), average='weighted', zero_division=0)
            del g_preds, g_labels, all_preds_t, all_labels_t
        else:
            t_prec, t_rec = t_acc, t_acc

        # ── VALIDATE ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                with autocast(enabled=(device.type == 'cuda')):
                    out = model(imgs)
                    loss = criterion(out, lbls)
                bs = imgs.size(0)
                v_loss += loss.item() * bs
                _, pred = out.max(1)
                v_correct += (pred == lbls).sum().item()
                v_total += bs
                if compute_full_metrics:
                    v_preds.append(pred)
                    v_labels.append(lbls)

        vl = reduce_tensor(torch.tensor(v_loss / v_total, device=device), world_size).item()
        va = reduce_tensor(torch.tensor(v_correct / v_total, device=device), world_size).item()

        if compute_full_metrics and v_preds:
            v_preds_t = torch.cat(v_preds)
            v_labels_t = torch.cat(v_labels)
            gv_preds, gv_labels = gather_predictions(v_preds_t, v_labels_t, world_size, device)
            v_prec, v_rec, _, _ = precision_recall_fscore_support(gv_labels.get(), gv_preds.get(), average='weighted', zero_division=0)
            del gv_preds, gv_labels, v_preds_t, v_labels_t
        else:
            v_prec, v_rec = va, va

        if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(vl)
        else: scheduler.step()

        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(vl); history['val_acc'].append(va)

        is_best = early_stop(vl, model)
        if is_best:
            best_vl, best_va, best_epoch = vl, va, epoch + 1
            if is_main:
                u, c = save_checkpoint(model, class_names, img_size, epoch + 1, vl, va, t_loss, t_acc, timestamp, model_name='ResNet50')
                logger.info(f"   💾 Best checkpoint → {Path(u).name}")

        ep_time = time.time() - ep_start
        if is_main:
            if compute_full_metrics:
                logger.info(f"└─ Train Loss={t_loss:.4f} Acc={t_acc:.2%} P={t_prec:.4f} R={t_rec:.4f} | Val Loss={vl:.4f} Acc={va:.2%} P={v_prec:.4f} R={v_rec:.4f} | {ep_time:.1f}s{' ⭐' if is_best else ''}")
            else:
                logger.info(f"└─ Train Loss={t_loss:.4f} Acc={t_acc:.2%} | Val Loss={vl:.4f} Acc={va:.2%} | {ep_time:.1f}s{' ⭐' if is_best else ''} (light)")

        if early_stop.early_stop:
            if is_main: logger.info(f"\n🛑 Early stopping at epoch {epoch+1}")
            raw = model.module if isinstance(model, DDP) else model
            raw.load_state_dict(early_stop.best_state)
            break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - start
    if not early_stop.early_stop and history['val_loss']:
        best_epoch = int(cp.argmin(history['val_loss'])) + 1
        best_vl = float(cp.min(history['val_loss']))
        best_va = history['val_acc'][best_epoch - 1]

    if is_main:
        logger.info(f"\n{'='*70}\n✅ Training done in {fmt_time(total_time)}\n   Best val_loss={best_vl:.6f} | val_acc={best_va:.2%} @ epoch {best_epoch}\n{'='*70}")

    return history, total_time, best_epoch, best_vl, best_va

# ============================================================================
# 🔍 SECTION 11: EVALUATION
# ============================================================================
def evaluate(model, test_loader, class_names, device, rank, world_size, logger):
    is_main = is_main_process(rank)
    if is_main: logger.info("\n[🔍 STEP 4] Evaluating on test set...")
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            with autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                probs = F.softmax(out, dim=1)
                _, pred = out.max(1)
            all_preds.append(pred)
            all_labels.append(lbls)
            all_probs.append(probs)

    all_preds_t = torch.cat(all_preds)
    all_labels_t = torch.cat(all_labels)
    all_probs_t = torch.cat(all_probs)
    g_preds, g_labels = gather_predictions(all_preds_t, all_labels_t, world_size, device)
    g_probs = gather_probs(all_probs_t, world_size, device)

    infer_time = time.time() - t0
    n = len(g_preds)
    fps = n / max(infer_time, 1e-9)
    acc = float(cp.mean(g_preds == g_labels))
    prec, rec, f1, _ = precision_recall_fscore_support(g_labels.get(), g_preds.get(), average='weighted', zero_division=0)
    cm = confusion_matrix(g_labels.get(), g_preds.get()) if is_main else None

    if is_main:
        logger.info(f"\n{'='*60}\nTEST RESULTS\n{'='*60}")
        logger.info(f"  Images:    {n:,}\n  Time:      {infer_time:.2f}s  ({fps:.1f} FPS)\n  Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n  Precision: {prec:.4f}\n  Recall:    {rec:.4f}\n  F1-Score:  {f1:.4f}\n{'='*60}")

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
        plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.close()
        logger.info("✓ confusion_matrix.png saved")

        rpt = classification_report(g_labels.get(), g_preds.get(), target_names=class_names, output_dict=True)
        for cn, m in rpt.items():
            if cn not in ('accuracy', 'macro avg', 'weighted avg'):
                logger.info(f"  {cn:20s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1-score']:.4f}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'inference_fps': fps, 'inference_time': infer_time, 'confusion_matrix': cm, 'predictions': g_preds, 'labels': g_labels, 'probabilities': g_probs}

# ============================================================================
# 📈 SECTION 12: PLOT TRAINING HISTORY
# ============================================================================
def plot_history(history, ts, best_ep, logger):
    Path("plots").mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_acc'], lw=2, label='Train')
    ax1.plot(history['val_acc'], lw=2, label='Val')
    ax1.axvline(best_ep - 1, color='r', ls='--', alpha=.7, label=f'Best (ep {best_ep})')
    ax1.set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy'); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(history['train_loss'], lw=2, label='Train')
    ax2.plot(history['val_loss'], lw=2, label='Val')
    ax2.axvline(best_ep - 1, color='r', ls='--', alpha=.7)
    ax2.scatter([best_ep - 1], [min(history['val_loss'])], color='r', s=100, zorder=5)
    ax2.set(title='Loss', xlabel='Epoch', ylabel='Loss'); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f'plots/training_history_{ts}.png', dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"✓ plots/training_history_{ts}.png saved")

# ============================================================================
# 🏁 SECTION 13: MAIN (✨ แก้ไขลำดับ DDP และ Optimizer แล้ว)
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Mango Pest Detection (ResNet50 + Discriminative LR)')
    parser.add_argument('--data_path', type=str, default='Dataset')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=500000)
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Base LR (สำหรับ classifier)')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='อัตราการลด LR สำหรับ layer ก่อนหน้า')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--aug_level', type=str, default='medium', choices=['light', 'medium', 'strong'])
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--use_cutmix', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--cutmix_prob', type=float, default=0.5)
    parser.add_argument('--disable_aug_epochs', type=int, default=20)
    parser.add_argument('--metrics_every_n', type=int, default=5)
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--use_discriminative_lr', action='store_true', default=True)
    args = parser.parse_args()

    rank, local_rank, world_size, is_dist = setup_distributed()
    logger = setup_logger(rank)
    is_main = is_main_process(rank)

    seed = 42
    torch.manual_seed(seed); cp.random.seed(seed); random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    cuda_ok, cuda_info, device = verify_cuda(local_rank)

    if is_main:
        logger.info("=" * 70 + "\nMULTI-GPU DDP PIPELINE — 🏆 RESNET50 + DISCRIMINATIVE LR\n" + "=" * 70)
        logger.info(f"  Time:        {datetime.now():%Y-%m-%d %H:%M:%S}\n  PyTorch:     {torch.__version__}\n  CUDA:        {torch.version.cuda or 'N/A'}\n  World Size:  {world_size} GPU(s)\n  Distributed: {'✅ DDP (NCCL)' if is_dist else '❌ Single process'}")
        for i in range(world_size):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"    GPU {i}: {p.name} ({p.total_memory / 1024**3:.1f} GB)")
        logger.info(f"  Batch/GPU:   {args.batch_size}  (effective = {args.batch_size * world_size})\n  Image Size:  {args.img_size}x{args.img_size}\n  Epochs:      {args.epochs}\n  Base LR:     {args.learning_rate}")
        logger.info(f"  🧠 Model:     ResNet50 {'(Pre-trained)' if args.pretrained else '(Random init)'}")
        logger.info(f"  🎯 Discriminative LR: {'✅' if args.use_discriminative_lr else '❌'} (decay={args.lr_decay})")
        logger.info(f"  🎨 Aug Level: {args.aug_level}")
        logger.info(f"  🎨 Mixup:     {'✅' if args.use_mixup else '❌'} (α={args.mixup_alpha})")
        logger.info(f"  🎨 CutMix:    {'✅' if args.use_cutmix else '❌'} (α={args.cutmix_alpha}, p={args.cutmix_prob})")
        logger.info(f"  ⚖️  Class Weights: {'✅' if args.use_class_weights else '❌'}")
        logger.info(f"  🐢 CPU Mode:  LOW (workers={args.num_workers}, metrics every {args.metrics_every_n} epochs)")
        logger.info("=" * 70)

    if is_main: logger.info("\n[📦 STEP 1] Loading dataset...")
    pipeline = DataPipeline(img_size=args.img_size, num_workers=args.num_workers, aug_level=args.aug_level)
    train_loader, val_loader, test_loader, class_names, train_sampler, val_sampler, test_sampler = pipeline.load_datasets(args.data_path, args.batch_size, is_dist, world_size, rank)
    num_classes = len(class_names)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if is_main:
        logger.info(f"  Classes ({num_classes}): {class_names}\n  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")
        with open("class_names.txt", 'w', encoding='utf-8') as f:
            for c in class_names: f.write(c + '\n')
        logger.info("  ✓ class_names.txt saved")

    # ✨ คำนวณ Class Weights
    class_weights = None
    if args.use_class_weights:
        if is_main: logger.info("\n[⚖️  STEP 1.5] Calculating class weights...")
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'samples'):
            class_counts = [0] * num_classes
            for _, label in train_dataset.samples:
                class_counts[label] += 1
        else:
            class_counts = [0] * num_classes
            for _, labels in train_loader:
                for label in labels:
                    class_counts[label.item()] += 1
            class_counts = [c * world_size for c in class_counts]
        total_samples = sum(class_counts)
        class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in class_counts]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        if is_main:
            logger.info(f"  Class counts: {class_counts}")
            for i, (cn, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
                logger.info(f"    {cn:20s}: {count:5d} samples → weight = {weight:.4f}")

    if is_main: logger.info(f"\n[🧠 STEP 2] Building ResNet50 ({num_classes} classes)...")
    
    # ✨ 1. สร้าง Raw Model ก่อน (ยังไม่ Wrap DDP)
    raw_model = build_resnet50(num_classes=num_classes, pretrained=args.pretrained).to(device)
    
    if is_main:
        logger.info(f"  Parameters:  {sum(p.numel() for p in raw_model.parameters()):,}")

    if args.use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # ✨ 2. ตั้งค่า Optimizer โดยใช้ raw_model (แก้ปัญหา AttributeError)
    if args.use_discriminative_lr:
        if is_main: logger.info("\n[🎯 STEP 2.5] Setting up Discriminative Learning Rates...")
        param_groups = get_parameter_groups(raw_model, base_lr=args.learning_rate, lr_decay=args.lr_decay)
        optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True)
        
        if is_main:
            logger.info("  📊 Learning Rate per Layer:")
            for g in param_groups:
                num_params = sum(p.numel() for p in g['params'])
                logger.info(f"    {g['name']:10s}: LR={g['lr']:.2e} | Params={num_params:,}")
    else:
        optimizer = optim.SGD(raw_model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)

    # ✨ 3. Wrap ด้วย DDP เป็นขั้นตอนสุดท้ายก่อนเริ่มเทรน
    if is_dist: 
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main: logger.info(f"  DDP wrapped: ✅")
    else: 
        model = raw_model
        if is_main: logger.info(f"  DDP wrapped: ❌")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history, t_time, best_ep, best_vl, best_va = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, ts, class_names, args.img_size,
        rank, world_size, is_dist, train_sampler, logger,
        use_mixup=args.use_mixup, use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, cutmix_prob=args.cutmix_prob,
        disable_aug_epochs=args.disable_aug_epochs, metrics_every_n=args.metrics_every_n
    )

    results = evaluate(model, test_loader, class_names, device, rank, world_size, logger)

    if is_main:
        Path("checkpoints").mkdir(exist_ok=True)
        raw = model.module if is_dist else model
        final_path = f'checkpoints/final_model_{ts}.pt'
        torch.save({
            'model_state_dict': {k: v.cpu() for k, v in raw.state_dict().items()},
            'class_names': class_names, 'class_indices': {n: i for i, n in enumerate(class_names)},
            'img_size': args.img_size, 'num_classes': num_classes, 'best_epoch': best_ep,
            'best_val_loss': best_vl, 'best_val_acc': best_va, 'test_accuracy': results['accuracy'],
            'test_precision': results['precision'], 'test_recall': results['recall'], 'test_f1': results['f1_score'],
            'inference_fps': results['inference_fps'], 'timestamp': ts, 'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_architecture': 'ResNet50', 'world_size': world_size,
            'aug_level': args.aug_level, 'use_mixup': args.use_mixup, 'use_cutmix': args.use_cutmix,
            'use_class_weights': args.use_class_weights, 'class_weights': class_weights if class_weights else None,
            'pretrained': args.pretrained, 'use_discriminative_lr': args.use_discriminative_lr,
            'lr_decay': args.lr_decay,
        }, final_path)
        logger.info(f"\n✅ Final model → {final_path}")

        info_path = f'logs/training_info_{ts}.txt'
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"MANGO PEST DETECTION — TRAINING REPORT (ResNet50 + Discriminative LR)\n{'='*50}\nTimestamp:     {ts}\nPyTorch:       {torch.__version__}\nCUDA:          {torch.version.cuda or 'N/A'}\nGPUs:          {world_size}\n")
            for i in range(world_size):
                p = torch.cuda.get_device_properties(i)
                f.write(f"  GPU {i}:       {p.name} ({p.total_memory / 1024**3:.1f} GB)\n")
            f.write(f"Image Size:    {args.img_size}\nBatch/GPU:     {args.batch_size}\nEffective BS:  {args.batch_size * world_size}\nClasses:       {num_classes} ({', '.join(class_names)})\n")
            f.write(f"Model:         ResNet50 {'(Pre-trained)' if args.pretrained else '(Random init)'}\n")
            f.write(f"Discriminative LR: {'✅' if args.use_discriminative_lr else '❌'} (decay={args.lr_decay})\n")
            f.write(f"Class Weights: {'✅' if args.use_class_weights else '❌'}\n")
            if class_weights:
                f.write(f"  Weights:     {[f'{w:.4f}' for w in class_weights]}\n")
            f.write(f"Aug Level:     {args.aug_level}\nMixup:         {'✅' if args.use_mixup else '❌'} (α={args.mixup_alpha})\nCutMix:        {'✅' if args.use_cutmix else '❌'} (α={args.cutmix_alpha}, p={args.cutmix_prob})\n")
            f.write(f"Disable Aug:   Last {args.disable_aug_epochs} epochs\n")
            f.write(f"CPU Mode:      LOW (workers={args.num_workers}, metrics every {args.metrics_every_n} epochs)\n")
            f.write(f"Best Epoch:    {best_ep}\nBest Val Loss: {best_vl:.6f}\nBest Val Acc:  {best_va:.4f}\nTest Acc:      {results['accuracy']:.4f}\nTest Prec:     {results['precision']:.4f}\nTest Rec:      {results['recall']:.4f}\nTest F1:       {results['f1_score']:.4f}\nFPS:           {results['inference_fps']:.1f}\nTrain Time:    {fmt_time(t_time)}\n")
        logger.info(f"✅ Info → {info_path}")

        plot_history(history, ts, best_ep, logger)

        logger.info(f"\n📁 Checkpoints:")
        for i, fp in enumerate(sorted(Path("checkpoints").glob("*.pt")), 1):
            tag = " ← improving val_loss" if 'best_model_epoch' in fp.name else (" ← CURRENT BEST" if fp.name == 'best_model.pt' else (" ← final + test metrics" if 'final_model' in fp.name else ""))
            logger.info(f"   [{i:2d}] {fp.name}{tag}")

        logger.info(f"\n{'='*70}\n✅ ALL DONE — {world_size} GPU(s) | {fmt_time(t_time)}\n   Model:         ResNet50 + Discriminative LR\n   Best val_loss: {best_vl:.6f} @ epoch {best_ep}\n   Test accuracy: {results['accuracy']:.4f}\n   Inference:     {results['inference_fps']:.1f} FPS\n{'='*70}")

    cleanup_distributed()

if __name__ == '__main__':
    main()

    