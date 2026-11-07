# Personal Protective Equipment (PPE) Detection Using Semi-Supervised Learning with YOLOv9-c 👷‍♂️👷‍♀️

> A real-time Personal Protective Equipment (PPE) detection system based on YOLOv9-c that leverages Semi-Supervised Learning (SSL) to overcome labeled data limitations, featuring automated compliance evaluation using the IoPPE metric.

This project was developed by a team from **Universitas Multimedia Nusantara (UMN)**, Tangerang, Banten, Indonesia. The system is designed to monitor worker safety in construction and industrial environments in real-time (38.4 FPS on an NVIDIA Tesla T4) by detecting 12 PPE classes.

## ✨ Key Features

* **12-Class PPE Detection:** Detects various objects such as `person`, `helmet`, `safety-vest`, `face-mask`, `glasses`, `gloves`, `shoes`, and more.
* **Semi-Supervised Learning (SSL):** Utilizes Adaptive Pseudo-labeling based on per-class confidence thresholds to leverage 40% of unlabeled data.
* **Programmable Gradient Information (PGI) & GELAN:** Leverages the state-of-the-art YOLOv9-c architecture to prevent information bottlenecks and improve feature extraction.
* **Intersection over PPE Area (IoPPE):** A novel calculation metric (replacing conventional IoU) to map PPE to individual workers to evaluate compliance status (Fully Compliant, Partially Compliant, Non-Compliant).
* **Explainable AI (XAI):** Uses **EigenCAM** (without backward pass) to provide visual explanations of the model's attention on detected PPE regions, proving that the model learns the correct semantic areas.

## 📊 Dataset

The dataset used is a combination of the **SH17** benchmark dataset and 7 other public datasets (Roboflow, aiotthien, CPPE-5, etc.), which have been standardized into the YOLO format.

* **Total Images:** 22,033 images
* **Total Annotations:** 104,929 bounding boxes
* **Data Split:** 60% Labeled Data (Phase 1) | 40% Unlabeled Data (Phase 2 SSL) | + Separate Validation Set (4,775 images).

## 🚀 Methodology & Training Results

Model training is divided into two main phases executed on dual NVIDIA Tesla T4 GPUs using Distributed Data Parallel (DDP):

### Phase 1: Supervised Baseline Training
The model is trained purely on the 60% labeled dataset for 25 epochs.
* **mAP@0.5:** 0.8308
* **Precision:** 0.8596
* **Recall:** 0.7995

### Phase 2: Adaptive Pseudo-Label Generation and SSL Self-Training
The Phase 1 model is used to generate pseudo-labels on the 40% unlabeled data. Thresholds are dynamically set (e.g., strict for `helmet` [0.65], lenient for `shoes` [0.28]). The model is then fine-tuned for 20 epochs.
* **mAP@0.5:** 0.8368 (+0.0060)
* **mAP@0.5:0.95:** 0.5981 (+0.0132)
* **Precision:** 0.8649 (+0.0053)
* **Recall:** 0.8122 (+0.0127)

The highest improvements occurred in minority classes: `body-protection` (+0.038), `face-mask` (+0.023), and `glasses` (+0.010).

## 🛠️ Compliance State Machine (IoPPE)

Each worker (person) is evaluated based on the detection of `helmet` (mandatory) and `safety-vest` (mandatory) mapped using the following formula:
`IoPPE = Area(PPE ∩ Person) / Area(PPE)`

* 🟩 **Fully Compliant:** Wearing Helmet and Safety Vest.
* 🟧 **Partially Compliant:** Wearing only one of the mandatory items (Helmet or Vest).
* 🟥 **Non-Compliant:** Not wearing any mandatory items.

## ⚙️ Installation

Follow these steps to set up the project environment locally. Ensure you have a CUDA-compatible GPU (NVIDIA Tesla T4 or equivalent recommended) for training and inference.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git](https://github.com/BryanValleryan/PPE-Compliance-Detection
   cd PPE-Compliance-Detection
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   conda create -n yolov9-ppe python=3.9 -y
   conda activate yolov9-ppe
   ```
   *Or using venv:*
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. **Install PyTorch according to your CUDA version:**
   *(Example for CUDA 11.8)*
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download Pre-trained Weights (Optional):**
   Download the pre-trained YOLOv9-c model for fine-tuning or use the weights resulting from this project's training (save them in the `weights/` folder).


## 👥 Project Contributors

* **Kenny Valent Winalda Sembiring** 
* **Bryan Valleryan Alvonso** 
* **Vaurolla Az Zahra** 
* **Aisha Reissani Sopyan** 
* **Amanda Aliyah** 

## 📜 License & Attribution

This project uses the YOLOv9 model base from [WongKinYiu's official repository](https://github.com/WongKinYiu/yolov9). The primary dataset (SH17) was created by H. M. Ahmad and A. Rahimi (University of Windsor).
