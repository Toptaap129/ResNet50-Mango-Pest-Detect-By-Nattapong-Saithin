"""
Model Evaluation & Visualization Script (✨ ResNet50 + ONNX Support)
Usage:
  # ประเมินโมเดล PyTorch (.pt)
  python 5.Evaluate_model.py --checkpoint checkpoints/best_model-ResNet50.pt --data_path Dataset
  
  # ประเมินโมเดล ONNX (.onnx) ที่ผ่านการบีบอัด
  python 5.Evaluate_model.py --checkpoint onnx-models/best_model-ResNet50_prune50_fp16.onnx --data_path Dataset
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
    top_k_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARN] onnxruntime not found. ONNX evaluation will be disabled.")

# ============================================================================
# 🧠 MODEL ARCHITECTURES (Fallback for PyTorch)
# ============================================================================
class CUDAOptimizedCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 96):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ============================================================================
# 🔧 UTILITY FUNCTIONS
# ============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pytorch_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt.get('num_classes', len(ckpt.get('class_names', [])))
    img_size = ckpt.get('img_size', 224)
    class_names = ckpt.get('class_names', [])
    arch = ckpt.get('model_architecture', 'CUDAOptimizedCNN')
    
    print(f"[INFO] Detected architecture: {arch.upper()}")
    
    if 'resnet' in arch.lower() or 'pretrained' in arch.lower():
        model = build_resnet50(num_classes=num_classes).to(device)
    else:
        model = CUDAOptimizedCNN(num_classes=num_classes, img_size=img_size).to(device)
        
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")
    
    return model, class_names, img_size, num_classes, arch, ckpt

def load_onnx_model(onnx_path: str, device: torch.device):
    if not ONNX_AVAILABLE:
        raise ImportError("กรุณาติดตั้ง onnxruntime: pip install onnxruntime-gpu")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    print(f"[INFO] ONNX Providers: {providers}")
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # ✨ อ่าน Metadata ที่ฝังไว้ในไฟล์ ONNX
    meta = session.get_modelmeta().custom_metadata_map
    if not meta:
        raise ValueError("ไม่พบ metadata ในไฟล์ ONNX กรุณาสร้างไฟล์ ONNX ใหม่ด้วย Compressor v3")
    
    try:
        class_names = json.loads(meta.get('class_names', '[]'))
        img_size = int(meta.get('img_size', 224))
        num_classes = int(meta.get('num_classes', len(class_names)))
        arch = meta.get('architecture', 'ONNX')
    except Exception as e:
        raise ValueError(f"ไม่สามารถ parse metadata จาก ONNX ได้: {e}")
    
    print(f"[INFO] Detected architecture: {arch.upper()} (via ONNX Metadata)")
    print(f"[INFO] Model parameters: N/A (ONNX)")
    
    return session, class_names, img_size, num_classes, arch, {}

def load_model(path: str, device: torch.device):
    ext = Path(path).suffix.lower()
    if ext == '.onnx':
        return load_onnx_model(path, device)
    else:
        return load_pytorch_model(path, device)

def build_test_loader(data_path: str, img_size: int, class_names: List[str], batch_size: int = 64):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    root = Path(data_path)
    test_dir = root / 'test'
    if not test_dir.exists():
        test_dir = root / 'val' if (root / 'val').exists() else root
        
    ds = ImageFolder(str(test_dir), transform=transform)
    
    class_to_idx_ckpt = {name: idx for idx, name in enumerate(class_names)}
    valid_samples = []
    skipped_folders = []
    
    for path, _ in ds.samples:
        folder_name = Path(path).parent.name
        if folder_name in class_to_idx_ckpt:
            valid_samples.append((path, class_to_idx_ckpt[folder_name]))
        else:
            skipped_folders.append(folder_name)
            
    if skipped_folders:
        unique_skipped = sorted(set(skipped_folders))
        print(f"\n⚠️  WARNING: Skipped {len(skipped_folders)} samples from folders not in checkpoint classes:")
        print(f"   Missing folders: {unique_skipped}\n")
        
    if not valid_samples:
        test_folders = sorted(set(Path(p).parent.name for p, _ in ds.samples))
        raise ValueError(f"No valid samples found!\nCheckpoint classes: {class_names}\nTest folders: {test_folders}")
        
    ds.samples = valid_samples
    ds.imgs = valid_samples
    ds.targets = [idx for _, idx in valid_samples]
    ds.class_to_idx = class_to_idx_ckpt
    ds.classes = class_names
    
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=False
    )
    return loader, ds

@torch.no_grad()
def run_inference(model: Union[nn.Module, ort.InferenceSession], loader, device, is_onnx: bool):
    all_preds, all_labels, all_probs = [], [], []
    
    if is_onnx:
        input_name = model.get_inputs()[0].name
        inp_type = model.get_inputs()[0].type
        use_fp16 = 'float16' in inp_type
        
    for imgs, lbls in loader:
        if is_onnx:
            # ONNX Inference
            np_imgs = imgs.cpu().numpy()
            np_imgs = np_imgs.astype(np.float16) if use_fp16 else np_imgs.astype(np.float32)
            
            out = model.run(None, {input_name: np_imgs})[0]
            
            # Stable Softmax in NumPy
            exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
            probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
            pred = np.argmax(probs, axis=1)
            
            all_preds.append(torch.from_numpy(pred))
            all_labels.append(lbls)
            all_probs.append(torch.from_numpy(probs))
        else:
            # PyTorch Inference
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                probs = F.softmax(out, dim=1)
                _, pred = out.max(1)
                
            all_preds.append(pred.cpu())
            all_labels.append(lbls)
            all_probs.append(probs.cpu())
            
    return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_probs)

def compute_metrics(preds, labels, probs, class_names):
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    top5 = top_k_accuracy_score(labels, probs.numpy(), k=min(5, len(class_names))) if len(class_names) >= 5 else acc
    cm = confusion_matrix(labels, preds)
    rpt = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    
    return {
        'acc': acc, 'prec': p, 'rec': r, 'f1': f1, 'top5': top5,
        'cm': cm, 'report': rpt
    }

def plot_all(preds, labels, probs, class_names, metrics, ts: str):
    Path("plots").mkdir(exist_ok=True)
    
    # 1) Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'plots/cm_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2) Per-class F1
    classes = [k for k in metrics['report'] if k not in ('accuracy', 'macro avg', 'weighted avg')]
    f1s = [metrics['report'][c]['f1-score'] for c in classes]
    fig, ax = plt.subplots(figsize=(10, max(4, len(classes) * 0.4)))
    bars = ax.barh(classes, f1s, color='steelblue')
    ax.set(xlim=(0, 1), xlabel='F1-Score', title='Per-Class F1-Score')
    for b, v in zip(bars, f1s):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(f'plots/per_class_f1_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3) Prediction confidence distribution
    conf = probs.max(dim=1).values.numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf, bins=30, color='teal', edgecolor='black', alpha=0.8)
    ax.axvline(conf.mean(), color='r', ls='--', label=f'Mean={conf.mean():.3f}')
    ax.set(xlabel='Max Softmax Probability', ylabel='Count', title='Prediction Confidence Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/conf_dist_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4) Wrong predictions by class
    wrong = (preds != labels).numpy()
    wrong_per_cls = [wrong[labels.numpy() == i].sum() for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(10, max(4, len(classes) * 0.4)))
    ax.barh(class_names, wrong_per_cls, color='salmon')
    ax.set(xlabel='Number of Wrong Predictions', title='Misclassified Samples per Class')
    for i, v in enumerate(wrong_per_cls):
        ax.text(v + 0.1, i, str(v), va='center')
    plt.tight_layout()
    plt.savefig(f'plots/wrong_per_class_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 🏁 MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description='Model Evaluation (✨ PyTorch + ONNX Support)')
    ap.add_argument('--checkpoint', type=str, default='best_model-ResNet50.pt', help='Path to model checkpoint (.pt or .onnx)')
    ap.add_argument('--data_path', type=str, default='Dataset', help='Path to dataset (should have test/ or val/ folder)')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    args = ap.parse_args()
    
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
    print(f"\n[INFO] Loading model: {args.checkpoint}")
    model, class_names, img_size, num_classes, arch, ckpt = load_model(args.checkpoint, device)
    is_onnx = isinstance(model, ort.InferenceSession)
    
    print(f"  Architecture : {arch.upper()}")
    print(f"  Classes      : {num_classes}")
    print(f"  Image size   : {img_size}x{img_size}")
    if not is_onnx:
        print(f"  Best val_acc : {ckpt.get('best_val_acc', ckpt.get('val_acc', 'N/A'))}")
        
    print(f"\n[INFO] Loading test data from: {args.data_path}")
    loader, ds = build_test_loader(args.data_path, img_size, class_names, args.batch_size)
    print(f"  Test samples (after filter): {len(ds):,}")
    
    print("\n[INFO] Running inference...")
    start_time = time.time()
    preds, labels, probs = run_inference(model, loader, device, is_onnx=is_onnx)
    infer_time = time.time() - start_time
    fps = len(ds) / max(infer_time, 1e-9)
    print(f"[INFO] Inference time: {infer_time:.2f}s ({fps:.1f} FPS)")
    
    print("\n[INFO] Computing metrics...")
    metrics = compute_metrics(preds, labels, probs, class_names)
    
    # Generate timestamp for plots
    ts = ckpt.get('timestamp', 'eval') if not is_onnx else 'onnx_eval'
    
    print("\n" + "=" * 60)
    print(f" 📊 TEST RESULTS  (model: {Path(args.checkpoint).name})")
    print("=" * 60)
    print(f"  Backend      : {'ONNX Runtime' if is_onnx else 'PyTorch'}")
    print(f"  Architecture : {arch.upper()}")
    print(f"  Samples      : {len(ds):,}")
    print(f"  Accuracy     : {metrics['acc']:.4f}  ({metrics['acc'] * 100:.2f}%)")
    print(f"  Top-5 Acc    : {metrics['top5']:.4f}")
    print(f"  Precision    : {metrics['prec']:.4f}")
    print(f"  Recall       : {metrics['rec']:.4f}")
    print(f"  F1-Score     : {metrics['f1']:.4f}")
    print(f"  Inference    : {fps:.1f} FPS")
    print("=" * 60)
    
    print("\n📋 Per-class report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    
    print("\n[INFO] Plotting graphs...")
    plot_all(preds, labels, probs, class_names, metrics, ts)
    print("\n[OK] Saved plots →")
    for p in sorted(Path("plots").glob(f"*_{ts}.png")):
        print(f"   • {p}")
        
    # ✨ บันทึก evaluation report
    report_path = f'logs/eval_report_{ts}.txt'
    Path("logs").mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"EVALUATION REPORT\n{'='*50}\n")
        f.write(f"Model:         {args.checkpoint}\n")
        f.write(f"Backend:       {'ONNX Runtime' if is_onnx else 'PyTorch'}\n")
        f.write(f"Architecture:  {arch.upper()}\n")
        f.write(f"Timestamp:     {ts}\n")
        f.write(f"Device:        {device}\n")
        f.write(f"Image Size:    {img_size}x{img_size}\n")
        f.write(f"Test Samples:  {len(ds):,}\n")
        f.write(f"Batch Size:    {args.batch_size}\n")
        f.write(f"Inference:     {infer_time:.2f}s ({fps:.1f} FPS)\n")
        f.write(f"\n{'='*50}\nMETRICS\n{'='*50}\n")
        f.write(f"Accuracy:      {metrics['acc']:.4f}\n")
        f.write(f"Top-5 Acc:     {metrics['top5']:.4f}\n")
        f.write(f"Precision:     {metrics['prec']:.4f}\n")
        f.write(f"Recall:        {metrics['rec']:.4f}\n")
        f.write(f"F1-Score:      {metrics['f1']:.4f}\n")
        f.write(f"\n{'='*50}\nPER-CLASS REPORT\n{'='*50}\n")
        f.write(classification_report(labels.numpy(), preds.numpy(), target_names=class_names, zero_division=0))
        
    print(f"\n[OK] Saved report → {report_path}")

if __name__ == '__main__':
    main()