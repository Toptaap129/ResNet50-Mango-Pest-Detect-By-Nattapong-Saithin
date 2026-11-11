====================================================================
Project Name: ResNet50-Mango-Pest-Detect-By-Nattapong-Saithin
ระบบวิเคราะห์โรคและศัตรูพืชมะม่วง (ใบและผล) ด้วย AI
====================================================================
พัฒนาโดย: T-A-N (Top-Aum-Nut)
ผู้ให้แนวคิดและไอเดียโดย: Aum (อุ้ม) , Nut (นัท)
ออกแบบโค้ดสำหรับการฝึกฝนโมเดลโดย: นายณัฐพงษ์ สายถิ่น (Top - ต๊อบ)

1. ภาพรวมโครงสร้างระบบ
- Frontend: เว็บ UI สำหรับอัปโหลดภาพและแสดงผล (3.index.html)
- Database: ฐานข้อมูลคำแนะนำการรักษาและรายละเอียดโรค (DataBase.js)
- Backend API: Flask Server สำหรับประมวลผลภาพด้วย PyTorch (2.Backend-ResNet50.py)
- Training Pipeline: สคริปต์เทรนโมเดลแบบ Multi-GPU (1.Multi-Cuda-ResNet50.py)

2. โครงสร้างไฟล์ (File Structure)
📂 ResNet50-Mango-Pest-Detect-By-Nattapong-Saithin
├── 1.Multi-Cuda-ResNet50.py      # Training Pipeline (Multi-GPU DDP)
├── 2.Backend-ResNet50.py         # Flask Inference API Server
├── 3.index.html                  # Frontend Web UI
├── DataBase.js                   # Treatment & Disease Database
├── class_names.txt               # รายชื่อ 13 คลาส
├── best_model.pt                 # โมเดลที่ดีที่สุด (Current Best)
├── confusion_matrix.png          # กราฟ Confusion Matrix หลังเทรนจบ
├── api_server.log                # Log การทำงานของ Backend API
├── 📂 Dataset/                   # โฟลเดอร์เก็บชุดข้อมูลภาพ (Train/Val/Test)
├── 📂 checkpoints/               # เก็บ Checkpoint ย่อย (best_model_epoch_..., final_model_...)
├── 📂 plots/                     # เก็บกราฟ Training History
└── 📂 logs/                      # เก็บ Log การเทรน (cuda_training_..., training_info_...)

3. สถาปัตยกรรมโมเดล AI (🧠 Model Architecture)
- โมเดลหลัก: ResNet50 (Pre-trained จาก ImageNet)
- โมเดลสำรอง: CUDAOptimizedCNN (Legacy CNN สำหรับ backward compatibility)
- Input Size: 224 x 224 pixels (RGB)
- Output: 13 Classes (Softmax probabilities)
- Framework: PyTorch + CUDA (รองรับ GPU Acceleration)

4. คลาสที่รองรับ (13 Classes)
1. Alternaria-Fruit          8. Gall Midge-leaf
2. Anthracnose-Fruit         9. Healthy-Fruit
3. Anthracnose-leaf         10. Healthy-leaf
4. Bacterial Canker-leaf    11. Powdery Mildew-leaf
5. Black Mould Rot-Fruit    12. Sooty Mould-leaf
6. Cutting Weevil-leaf      13. Stem end Rot-Fruit
7. Die Back-leaf

5. เทคนิคการเทรนโมเดล (Training Techniques)
- Optimization:
  • Optimizer: SGD (momentum=0.9, nesterov=True)
  • Scheduler: CosineAnnealingLR
  • Discriminative Learning Rates (LR ต่างกันแต่ละ layer)
- Data Augmentation:
  • 3 Levels: light / medium / strong
  • Mixup (α=0.2) + CutMix (α=1.0, p=0.5)
  • ปิด Augmentation 20 epochs สุดท้ายก่อนจบเทรน
- Regularization:
  • Class Weights (คำนวณจาก inverse frequency)
  • Early Stopping (patience=20)
- Distributed Training:
  • รองรับ Single GPU / Multi-GPU (DDP + NCCL backend)
  • CuPy Zero-copy สำหรับ gather predictions

6. API Endpoints (Backend)
เซิร์ฟเวอร์รันที่: 0.0.0.0:5000
- POST /predict : รับภาพเพื่อทำนายโรค (ส่งไฟล์ผ่านคีย์ 'image')
- GET /health   : ตรวจสอบสถานะเซิร์ฟเวอร์และโมเดล
- GET /classes  : ดึงรายชื่อคลาสทั้งหมด

7. Preprocessing Pipeline (Inference)
- Resize → 224x224 (Bilinear)
- ToTensor
- Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

8. ฐานข้อมูลคำแนะนำ (DataBase.js)
จัดเก็บข้อมูลเชิงลึกของทั้ง 13 คลาส ได้แก่:
- การจำแนกประเภท (healthy / disease / pest)
- เชื้อก่อโรคหรือศัตรูพืช (pathogen / pest)
- อาการเบื้องต้น (symptoms)
- วิธีการรักษาและป้องกัน (treatment)

9. การรันระบบ
- เทรนโมเดล (Single GPU):
  python 1.Multi-Cuda-ResNet50.py --data_path Dataset
- เทรนโมเดล (Multi-GPU):
  torchrun --nproc_per_node=2 1.Multi-Cuda-ResNet50.py --data_path Dataset
- รัน Backend API:
  python 2.Backend-ResNet50.py
- เปิด Frontend: เปิดไฟล์ 3.index.html ในเบราว์เซอร์
====================================================================
