"""
🥭 MANGO PEST DETECTION - INFERENCE API (FLASK + ONNX/PYTORCH)
Production-Ready Inference Server - Auto-detect ONNX or PyTorch checkpoint
✨ v3: Reads metadata directly from ONNX file (no .pt dependency)
"""
import os
import sys
import io
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api_server.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model-ResNet50.pt")

inference_backend = None
model_session = None
class_names = []
checkpoint_info = {}
img_size = 224
num_classes = 0
device_name = "CPU"
architecture = "Unknown"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# ============================================================================
# 📋 METADATA EXTRACTION FROM ONNX
# ============================================================================
def extract_onnx_metadata(session):
    """✨ ดึง metadata ที่ฝังอยู่ในไฟล์ ONNX (class_names, img_size, etc.)"""
    metadata = {}
    try:
        meta_map = session.get_modelmeta().custom_metadata_map
        if not meta_map:
            return metadata
        
        logger.info("   📋 Reading embedded metadata from ONNX...")
        
        for key, value in meta_map.items():
            if key == 'class_names':
                try:
                    metadata['class_names'] = json.loads(value)
                except:
                    metadata['class_names'] = []
            elif key in ['img_size', 'num_classes']:
                try:
                    metadata[key] = int(value)
                except:
                    pass
            else:
                metadata[key] = value
        
        logger.info(f"      → class_names: {metadata.get('class_names', [])}")
        logger.info(f"      → img_size: {metadata.get('img_size', 224)}")
        logger.info(f"      → num_classes: {metadata.get('num_classes', 0)}")
        logger.info(f"      → architecture: {metadata.get('architecture', 'Unknown')}")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to read ONNX metadata: {e}")
    
    return metadata

# ============================================================================
# 🖼️ IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image: Image.Image, target_size: int) -> np.ndarray:
    img = image.resize((target_size, target_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return np.ascontiguousarray(img_array)

# ============================================================================
# 🚀 ONNX RUNTIME BACKEND
# ============================================================================
def load_onnx_model(path: str):
    global model_session, device_name, class_names, img_size, num_classes, architecture
    
    import onnxruntime as ort
    
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = min(4, os.cpu_count() or 1)
    so.inter_op_num_threads = 2
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        device_name = "CUDA (GPU)"
        logger.info("   → Using CUDAExecutionProvider (GPU)")
    else:
        providers = ['CPUExecutionProvider']
        device_name = "CPU"
        logger.info("   → Using CPUExecutionProvider")
    
    logger.info(f"   → Loading ONNX model: {path}")
    model_session = ort.InferenceSession(path, so, providers=providers)
    
    # ✨ ดึง metadata ที่ฝังอยู่ในไฟล์ ONNX
    embedded_meta = extract_onnx_metadata(model_session)
    if embedded_meta:
        class_names = embedded_meta.get('class_names', [])
        img_size = embedded_meta.get('img_size', 224)
        num_classes = embedded_meta.get('num_classes', len(class_names))
        architecture = embedded_meta.get('architecture', 'ONNX')
    
    input_info = model_session.get_inputs()[0]
    logger.info(f"   → Input: {input_info.name} {input_info.shape} ({input_info.type})")
    
    # 🔥 Warmup
    logger.info("   → Warming up inference engine...")
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(5):
        model_session.run(None, {input_info.name: dummy})
    logger.info("   ✅ Warmup complete")
    
    return True

# ============================================================================
# 🔥 PYTORCH BACKEND (Fallback)
# ============================================================================
def load_pytorch_model(path: str):
    global model_session, device_name, class_names, img_size, num_classes, architecture
    
    import torch
    import torch.nn as nn
    from torchvision import models
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else "CPU"
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    arch = checkpoint.get('model_architecture', 'CUDAOptimizedCNN')
    class_names = checkpoint.get('class_names', [])
    img_size = checkpoint.get('img_size', 224)
    num_classes = checkpoint.get('num_classes', len(class_names))
    architecture = arch
    
    logger.info(f"   → Architecture: {arch}")
    logger.info(f"   → Device: {device_name}")
    logger.info(f"   → Classes: {class_names}")
    
    if arch == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        class CUDAOptimizedCNN(nn.Module):
            def __init__(self, num_classes, img_size=224):
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
        model = CUDAOptimizedCNN(num_classes)
    
    sd = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    model_session = model
    
    logger.info("   → Warming up PyTorch inference...")
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    logger.info("   ✅ Warmup complete")
    
    return True

# ============================================================================
# 🎯 UNIFIED LOADER (✨ แก้ bug: เพิ่ม global declaration ครบทุกตัวแปร)
# ============================================================================
def load_model_checkpoint(path: str):
    # 🔧 FIX: Declare global ทุกตัวแปรที่จะถูก assign ในฟังก์ชันนี้
    global inference_backend, checkpoint_info, class_names, img_size, num_classes, architecture
    
    if not os.path.exists(path):
        logger.error(f"❌ Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")
    
    logger.info(f"🚀 Loading model from {path}...")
    ext = Path(path).suffix.lower()
    
    if ext == '.onnx':
        inference_backend = 'onnx'
        load_onnx_model(path)
    elif ext in ['.pt', '.pth']:
        inference_backend = 'pytorch'
        load_pytorch_model(path)
    else:
        raise ValueError(f"❌ Unsupported model format: {ext}")
    
    # Fallback: ถ้าไม่มี metadata เลย (เช่น ONNX เก่าที่ไม่มี metadata)
    if not class_names:
        logger.warning("   ⚠️ No class names found, using environment variables")
        env_classes = os.environ.get("CLASS_NAMES", "")
        if env_classes:
            class_names = [c.strip() for c in env_classes.split(',')]
            num_classes = len(class_names)
        else:
            class_names = [f"Class_{i}" for i in range(max(1, num_classes))]
    
    checkpoint_info = {
        'path': path,
        'filename': os.path.basename(path),
        'loaded_at': datetime.now().isoformat(),
        'backend': inference_backend,
        'architecture': architecture,
        'file_size_mb': round(os.path.getsize(path) / (1024*1024), 2)
    }
    
    logger.info(f"✅ Model loaded successfully via {inference_backend.upper()} backend")
    logger.info(f"   → Final class_names: {class_names}")
    return True

# ============================================================================
# 🔮 INFERENCE
# ============================================================================
def infer_onnx(input_array: np.ndarray):
    input_name = model_session.get_inputs()[0].name
    inp_type = model_session.get_inputs()[0].type
    if 'float16' in inp_type:
        input_array = input_array.astype(np.float16)
    else:
        input_array = input_array.astype(np.float32)
    outputs = model_session.run(None, {input_name: input_array})
    return outputs[0].astype(np.float32)

def infer_pytorch(input_array: np.ndarray):
    import torch
    import torch.nn.functional as F
    
    device = next(model_session.parameters()).device
    tensor = torch.from_numpy(input_array).to(device)
    with torch.no_grad():
        output = model_session(tensor)
        probs = F.softmax(output, dim=1)
    return probs.cpu().numpy()

# ============================================================================
# 🌐 API ENDPOINTS
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    if model_session is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_array = preprocess_image(image, img_size)
        
        start_time = time.time()
        if inference_backend == 'onnx':
            output = infer_onnx(input_array)
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        else:
            probabilities = infer_pytorch(input_array)
        
        inference_time = time.time() - start_time
        
        probs_np = probabilities[0]
        pred_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx])
        
        safe_classes = class_names if len(class_names) >= len(probs_np) else \
                       class_names + [f"Class_{i}" for i in range(len(class_names), len(probs_np))]
        
        predictions_list = [
            {'class': safe_classes[i], 'probability': round(float(p), 4)}
            for i, p in enumerate(probs_np)
        ]
        predictions_list.sort(key=lambda x: x['probability'], reverse=True)
        
        response = {
            'success': True,
            'prediction': {
                'class': safe_classes[pred_idx],
                'confidence': round(confidence, 4),
                'index': pred_idx
            },
            'all_probabilities': predictions_list,
            'inference_time_ms': round(inference_time * 1000, 2),
            'model_info': {
                'path': checkpoint_info.get('filename'),
                'backend': inference_backend,
                'device': device_name,
                'architecture': architecture,
                'file_size_mb': checkpoint_info.get('file_size_mb')
            }
        }
        
        logger.info(f"🔮 [{inference_backend.upper()}] {response['prediction']['class']} "
                   f"({confidence:.2%}) in {inference_time*1000:.1f}ms")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error during prediction',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    is_model_loaded = model_session is not None
    return jsonify({
        'status': 'healthy' if is_model_loaded else 'degraded',
        'service': 'Mango Pest Detection API',
        'backend': inference_backend,
        'device': device_name,
        'model_loaded': is_model_loaded,
        'model_info': checkpoint_info if is_model_loaded else None,
        'num_classes': num_classes,
        'class_names': class_names,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'num_classes': num_classes,
        'class_names': class_names
    }), 200

# ============================================================================
# 🏁 MAIN
# ============================================================================
if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🥭 MANGO PEST DETECTION API SERVER STARTING")
    logger.info("=" * 80)
    
    try:
        load_model_checkpoint(MODEL_PATH)
        logger.info("-" * 80)
        logger.info("🌍 Server Configuration:")
        logger.info(f"   Host: 0.0.0.0")
        logger.info(f"   Port: 5000")
        logger.info(f"   Model: {MODEL_PATH}")
        logger.info(f"   Backend: {inference_backend.upper()}")
        logger.info(f"   Device: {device_name}")
        logger.info(f"   Classes: {class_names}")
        logger.info("-" * 80)
        logger.info("✅ Server Ready to Accept Requests")
        logger.info("=" * 80)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        logger.critical(f"🚫 Failed to start server: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)