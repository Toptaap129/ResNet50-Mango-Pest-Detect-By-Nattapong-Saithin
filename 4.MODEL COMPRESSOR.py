"""
============================================================================
🗜️ MODEL COMPRESSOR PRO v3: Compression + Metadata Embedding
============================================================================
บีบอัดโมเดล + ฝัง metadata (class_names, img_size) ลงในไฟล์ ONNX โดยตรง
รองรับ: ResNet50, CUDAOptimizedCNN

Usage:
  python "4.MODEL COMPRESSOR.py" --model best_model-ResNet50.pt --method prune50
============================================================================
"""
import os, sys, time, json, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path

try:
    import onnx
    import onnxruntime as ort
    from onnxsim import simplify
except ImportError:
    print("❌ กรุณาติดตั้ง: pip install onnx onnxruntime onnxsim")
    sys.exit(1)

from torchvision import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 🧠 MODEL ARCHITECTURES
# ============================================================================
class CUDAOptimizedCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(architecture, num_classes, img_size=224):
    if architecture == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return CUDAOptimizedCNN(num_classes, img_size)

def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    arch = ckpt.get('model_architecture', 'CUDAOptimizedCNN')
    meta = {
        'architecture': arch,
        'class_names': ckpt.get('class_names', []),
        'img_size': ckpt.get('img_size', 224),
        'num_classes': ckpt.get('num_classes', len(ckpt.get('class_names', []))),
        'val_acc': ckpt.get('val_acc'),
        'epoch': ckpt.get('epoch'),
    }
    model = build_model(arch, meta['num_classes'], meta['img_size'])
    sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()
    return model, meta

# ============================================================================
# ✂️ PRUNING ONLY
# ============================================================================
def prune_model(model, ratio):
    logger.info(f"\n✂️ Pruning {ratio*100:.0f}%...")
    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' not in name:
            params_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=ratio,
    )
    for module, _ in params_to_prune:
        prune.remove(module, 'weight')
    
    total_after = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters หลัง prune: {total_after:,}")
    return model

# ============================================================================
# 🔄 EXPORT + CONVERT + METADATA
# ============================================================================
def export_onnx(model, img_size, output_path, opset=17):
    model = model.cpu()
    if hasattr(model, 'module'):
        model = model.module
        
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True, opset_version=opset,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    try:
        m = onnx.load(output_path)
        s, ok = simplify(m)
        if ok: onnx.save(s, output_path)
    except Exception as e:
        logger.warning(f"   Simplify failed: {e}")
    
    size = os.path.getsize(output_path) / (1024*1024)
    logger.info(f"   ✅ ONNX FP32: {size:.2f} MB")
    return output_path

def attach_metadata(onnx_path, metadata):
    """✨ แนบ metadata (class_names, img_size, etc.) ลงในไฟล์ ONNX โดยตรง"""
    try:
        m = onnx.load(onnx_path)
        
        # ONNX metadata_props เก็บเป็น string → string เท่านั้น
        # ต้อง serialize dict/list เป็น JSON string
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                str_value = json.dumps(value, ensure_ascii=False)
            else:
                str_value = str(value)
            
            entry = onnx.StringStringEntryProto()
            entry.key = key
            entry.value = str_value
            m.metadata_props.append(entry)
        
        onnx.save(m, onnx_path)
        logger.info(f"   📋 Embedded metadata: {list(metadata.keys())}")
        logger.info(f"      → class_names: {metadata.get('class_names', [])}")
        logger.info(f"      → img_size: {metadata.get('img_size')}")
        logger.info(f"      → num_classes: {metadata.get('num_classes')}")
        logger.info(f"      → architecture: {metadata.get('architecture')}")
        return True
    except Exception as e:
        logger.error(f"   ❌ Failed to attach metadata: {e}")
        return False

def convert_to_fp16(src, dst):
    try:
        from onnxconverter_common import float16
        m = onnx.load(src)
        m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m_fp16, dst)
        size = os.path.getsize(dst) / (1024*1024)
        logger.info(f"   ✅ FP16: {size:.2f} MB")
        return dst
    except ImportError:
        logger.error("   ❌ กรุณาติดตั้ง: pip install onnxconverter-common")
        return src
    except Exception as e:
        logger.error(f"   ❌ FP16 Conversion failed: {e}")
        return src

def quantize_int8(src, dst):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            src, dst, weight_type=QuantType.QUInt8,
            extra_options={'WeightSymmetric': False, 'MatMulConstBOnly': True}
        )
        size = os.path.getsize(dst) / (1024*1024)
        logger.info(f"   ✅ INT8: {size:.2f} MB")
        return dst
    except Exception as e:
        logger.error(f"   ❌ INT8 Quantization failed: {e}")
        return src

# ============================================================================
# ✅ VERIFY
# ============================================================================
def verify(pt_model, onnx_path, img_size):
    pt_model = pt_model.cpu()
    if hasattr(pt_model, 'module'):
        pt_model = pt_model.module
        
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(dummy)).numpy()
    
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        sess = ort.InferenceSession(onnx_path, so, providers=providers)
    except Exception as e:
        logger.error(f"   ❌ Load failed: {e}")
        return None
    
    inp = sess.get_inputs()[0]
    onnx_out = sess.run(None, {inp.name: dummy})[0].astype(np.float32)
    
    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    argmax_ok = int(np.argmax(pt_out)) == int(np.argmax(onnx_out))
    
    t0 = time.time()
    for _ in range(50): sess.run(None, {inp.name: dummy})
    ms = (time.time() - t0) / 50 * 1000
    
    logger.info(f"   Max diff: {max_diff:.4f} | Argmax: {'✅' if argmax_ok else '❌'} | {ms:.1f} ms")
    return {'max_diff': max_diff, 'argmax_match': argmax_ok, 'latency_ms': ms}

# ============================================================================
# 🏁 MAIN
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='best_model-ResNet50.pt')
    p.add_argument('--output_dir', default='onnx-models')
    p.add_argument('--method', default='prune50',
                   choices=['int8', 'prune25', 'prune50', 'prune70'])
    args = p.parse_args()
    
    logger.info("="*70)
    logger.info("🗜️ MODEL COMPRESSOR PRO v3 - With Metadata Embedding")
    logger.info("="*70)
    
    orig_size = os.path.getsize(args.model) / (1024*1024)
    logger.info(f"Original: {orig_size:.2f} MB")
    
    pt_model, meta = load_checkpoint(args.model)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.model).stem
    base = f"{stem}_{args.method}"
    
    # 1. Prune (ถ้ามีใน method)
    if 'prune' in args.method:
        ratio = 0.25 if '25' in args.method else (0.50 if '50' in args.method else 0.70)
        pt_model = prune_model(pt_model, ratio)
    
    # 2. Export ONNX FP32
    fp32_path = os.path.join(args.output_dir, f"{base}_fp32.onnx")
    export_onnx(pt_model, meta['img_size'], fp32_path)
    
    # 3. ✨ แนบ metadata ลงใน FP32 ก่อน (จะติดไปกับ FP16/INT8 ด้วย)
    metadata_to_embed = {
        'class_names': meta['class_names'],
        'img_size': meta['img_size'],
        'num_classes': meta['num_classes'],
        'architecture': meta['architecture'],
        'source_model': os.path.basename(args.model),
        'compression_method': args.method,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if meta.get('val_acc') is not None:
        metadata_to_embed['original_val_acc'] = meta['val_acc']
    
    attach_metadata(fp32_path, metadata_to_embed)
    
    # 4. Convert/Quantize (metadata จะถูก preserve ไว้)
    if 'prune' in args.method:
        final_path = os.path.join(args.output_dir, f"{base}_fp16.onnx")
        convert_to_fp16(fp32_path, final_path)
    else:
        final_path = os.path.join(args.output_dir, f"{base}_int8.onnx")
        quantize_int8(fp32_path, final_path)
        
    if os.path.exists(fp32_path): os.remove(fp32_path)
    
    # 5. Verify
    metrics = verify(pt_model, final_path, meta['img_size'])
    
    # 6. ✨ ตรวจสอบว่า metadata ติดไปด้วย
    logger.info("\n🔍 Verifying embedded metadata in final ONNX...")
    try:
        sess = ort.InferenceSession(final_path, providers=['CPUExecutionProvider'])
        embedded_meta = sess.get_modelmeta().custom_metadata_map
        if embedded_meta:
            logger.info(f"   ✅ Metadata found in final ONNX:")
            for k, v in embedded_meta.items():
                if k == 'class_names':
                    logger.info(f"      → {k}: {json.loads(v)}")
                else:
                    logger.info(f"      → {k}: {v}")
        else:
            logger.warning("   ⚠️ No metadata found in final ONNX")
    except Exception as e:
        logger.warning(f"   ⚠️ Could not verify metadata: {e}")
    
    size = os.path.getsize(final_path) / (1024*1024)
    reduction = (1 - size/orig_size) * 100
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 RESULTS: {args.method}")
    logger.info(f"{'='*70}")
    logger.info(f"Size:      {size:.2f} MB ({reduction:+.1f}%)")
    if metrics:
        logger.info(f"Max diff:  {metrics['max_diff']:.4f}")
        logger.info(f"Argmatch:  {'✅' if metrics['argmax_match'] else '❌'}")
        logger.info(f"Latency:   {metrics['latency_ms']:.1f} ms")
    
    logger.info(f"\n💡 ใช้งานใน Backend (ไฟล์เดียวจบ!):")
    logger.info(f"   MODEL_PATH={final_path} python 2.Backend-ResNet50.py")

if __name__ == '__main__':
    main()