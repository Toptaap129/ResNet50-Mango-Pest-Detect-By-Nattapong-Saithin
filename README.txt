====================================================================
AI-Based Web Analysis of Mango Leaves and Fruits
====================================================================
ระบบวิเคราะห์โรคและศัตรูพืชมะม่วง (ใบและผล) ด้วย AI
พัฒนาโดย: T-A-N (Top-Aum-Nut)
ออกแบบโค้ดสำหรับการฝึกฝนโมเดลโดย: นายณัฐพงษ์ สายถิ่น (Top)

1. ภาพรวมโครงสร้างระบบ
- Frontend: เว็บ UI สำหรับอัปโหลดภาพและแสดงผล (3.index.html)
- Database: ฐานข้อมูลคำแนะนำการรักษาและรายละเอียดโรค (DataBase.js)
- Backend API: Flask Server สำหรับประมวลผลภาพด้วย PyTorch (2.Backend-ResNet50.py)
- Training Pipeline: สคริปต์เทรนโมเดลแบบ Multi-GPU (1.Multi-Cuda-ResNet50.py)

2. สถาปัตยกรรมโมเดล AI (🧠 Model Architecture)
- โมเดลหลัก: ResNet50 (Pre-trained จาก ImageNet)
- โมเดลสำรอง: CUDAOptimizedCNN (Legacy CNN สำหรับ backward compatibility)
- Input Size: 224 x 224 pixels (RGB)
- Output: 13 Classes (Softmax probabilities)
- Framework: PyTorch + CUDA (รองรับ GPU Acceleration)

3. คลาสที่รองรับ (13 Classes)
1. Alternaria-Fruit          8. Gall Midge-leaf
2. Anthracnose-Fruit         9. Healthy-Fruit
3. Anthracnose-leaf         10. Healthy-leaf
4. Bacterial Canker-leaf    11. Powdery Mildew-leaf
5. Black Mould Rot-Fruit    12. Sooty Mould-leaf
6. Cutting Weevil-leaf      13. Stem end Rot-Fruit
7. Die Back-leaf

4. เทคนิคการเทรนโมเดล (Training Techniques)
- Optimization:
  • Optimizer: SGD (momentum=0.9, nesterov=True)
  • Scheduler: CosineAnnealingLR
  • Discriminative Learning Rates (LR ต่างกันแต่ละ layer)
    - conv1+bn1 : LR ต่ำสุด (base_lr × decay³)
    - layer1-4  : LR เพิ่มขึ้นตามความลึก
    - fc (classifier): LR สูงสุด (base_lr)
- Data Augmentation:
  • 3 Levels: light / medium / strong
  • Mixup (α=0.2) + CutMix (α=1.0, p=0.5)
  • ปิด Augmentation 20 epochs สุดท้ายก่อนจบเทรน
- Regularization:
  • Class Weights (คำนวณจาก inverse frequency)
  • Early Stopping (patience=20)
  • Dropout ใน Legacy CNN
- Distributed Training:
  • รองรับ Single GPU / Multi-GPU (DDP + NCCL backend)
  • CuPy Zero-copy สำหรับ gather predictions

5. API Endpoints (Backend)
เซิร์ฟเวอร์รันที่: 0.0.0.0:5000
- POST /predict : รับภาพเพื่อทำนายโรค (ส่งไฟล์ผ่านคีย์ 'image')
                   คืนค่า class, confidence, all_probabilities
- GET /health   : ตรวจสอบสถานะเซิร์ฟเวอร์และโมเดล
- GET /classes  : ดึงรายชื่อคลาสทั้งหมด

6. Preprocessing Pipeline (Inference)
- Resize → 224x224 (Bilinear)
- ToTensor
- Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

7. ฐานข้อมูลคำแนะนำ (DataBase.js)
จัดเก็บข้อมูลเชิงลึกของทั้ง 13 คลาส ได้แก่:
- การจำแนกประเภท (healthy / disease / pest)
- เชื้อก่อโรคหรือศัตรูพืช (pathogen / pest)
- อาการเบื้องต้น (symptoms)
- วิธีการรักษาและป้องกัน (treatment)

8. ไฟล์ Checkpoint ที่สร้างจากการเทรน
- best_model.pt              → โมเดลที่ดีที่สุดปัจจุบัน
- best_model_epoch_XXX.pt    → โมเดลแต่ละ epoch ที่ val_loss ดีขึ้น
- final_model_TIMESTAMP.pt   → โมเดลสุดท้ายพร้อม test metrics
- class_names.txt            → รายชื่อคลาส
- confusion_matrix.png       → เมทริกซ์ความสับสน
- plots/training_history.png → กราฟ loss/accuracy

9. การรันระบบ
- เทรนโมเดล (Single GPU):
  python 1.Multi-Cuda-ResNet50.py --data_path Dataset
- เทรนโมเดล (Multi-GPU):
  torchrun --nproc_per_node=2 1.Multi-Cuda-ResNet50.py --data_path Dataset
- รัน Backend API:
  python 2.Backend-ResNet50.py
- เปิด Frontend: เปิดไฟล์ 3.index.html ในเบราว์เซอร์
====================================================================
