# 🚗 Enhanced SparseDrive with Latent World Model (LAW-DINO Integration)

> **Research Abstract:** 
> 
> This project explores a potential approach to a challenge in autonomous driving: integrating predictive capabilities into perception-based driving systems. While existing end-to-end architectures appear effective for reactive decision-making, they may lack capabilities to anticipate future scene evolution. We propose a framework that aims to augment the SparseDrive architecture with a self-supervised Latent World Model (LWM) trained on DINOv2 embeddings, potentially enabling more temporally coherent scene understanding.
>
> Our approach investigates learning action-conditioned dynamics in a latent space without requiring additional annotations. Preliminary evaluation on the NuScenes-Mini dataset suggests potential improvements: initial results indicate approximately 62% reduction in collision rate, 40% fewer tracking ID switches, and 14% lower trajectory error compared to our baseline implementation. These early findings may indicate that learned latent dynamics models could help enhance safety-critical metrics in autonomous driving systems, representing a potentially promising direction for bridging perception and control in embodied intelligence that warrants further investigation.

## 🌟 Overview
This repository aims to extend **SparseDrive** -- a sparse, end-to-end autonomous driving framework -- by exploring the integration of a **Latent World Model (LWM)** branch inspired by *LAW (Enhancing End-to-End Autonomous Driving with Latent World Model, ICLR 2025)*.  
Rather than training a visual encoder from scratch, we experiment with a **frozen DINOv2** backbone with the goal of leveraging its potentially rich, geometry-aware features.  
The model attempts to **predict the evolution of scene representations in latent space**, conditioned on **ego actions**, with the aim of potentially improving temporal consistency and planning robustness without requiring extra supervision.

---

## 🧭 Philosophy
Many traditional end-to-end driving pipelines appear to operate primarily in a *reactive* manner: processing each frame independently, with the planner responding primarily to the current state.  
Our proposed **Latent World Model** approach aims to introduce more *predictive reasoning* -- attempting to learn how the latent scene state might change given the vehicle's motion.

Our proposed formulation:

$$\hat{z}_{t+1}=f_{\theta}(z_t,u_t),\quad z_t=g(\text{DINO}(x_t))$$

Here  
- $z_t$: latent embedding extracted from DINOv2 features,  
- $u_t$: ego action or egomotion (steering, throttle, brake, Deltapose),  
- $f_{\theta}$: latent dynamics network that attempts to predict the next latent state.  

During training, we encourage the model to align its prediction $\hat{z}_{t+1}$ with the **frozen target latent** $z_{t+1}^{\text{tgt}}$ from the next frame.

We hypothesize that this predictive supervision approach might help the planner develop more temporally consistent representations. By attempting to learn the dynamics of scene evolution in latent space, our goal is to explore whether the model could potentially anticipate future states and make more informed planning decisions under uncertainty, which might help address what we believe is a limitation in some current autonomous driving systems.

---

## 🧠 Contributions
1. Integration of a **LAW-style latent world model** into SparseDrive's Stage-2 planning pipeline.  
2. **DINOv2-based latent encoder** providing strong pretrained semantics.  
3. **Action-conditioned latent dynamics** that model how visual states evolve.  
4. **Auxiliary world-model loss** improving temporal consistency and robustness.  
5. **Empirical gains** in tracking stability and planning safety on mini-NuScenes.
---

## ⚙️ Experimental Setup and Methodology

### Implementation Details
- **Stage-1:** Standard SparseDrive perception pretraining with detection and segmentation objectives
- **Stage-2:** Joint training of the planner and latent world model with temporal consistency constraints
- **Encoder:** `facebook/dinov2-large` (frozen) providing 1024-dimensional features
- **Dynamics Network:** MLP layers with residual connections and layer normalization (2.4M parameters)
- **Optimization:** AdamW (β₁=0.9, β₂=0.999), cosine LR schedule (initial lr=1e-4), λ<sub>wm</sub>=0.7
- **Batch configuration:** 4–8 sequential frames for temporal pairing, 16 scenes per batch
- **Regularization:** Weight decay=0.01, gradient clipping at norm=1.0, dropout=0.1  

---

## 🧮 Loss Functions and Optimization

### **World-Model Loss**
The auxiliary self-supervised objective combines cosine similarity and L2 distance between predicted and target latents:

$$\mathcal{L}_{wm}=\alpha\big(1-\cos(\hat{z}_{t+1},z_{t+1}^{tgt})\big)+\beta\lVert\hat{z}_{t+1}-z_{t+1}^{tgt}\rVert_2^2$$

The cosine term enforces structural similarity in the latent space, while the L2 term ensures metric accuracy. This dual objective helps balance semantic coherence with geometric precision in the predicted representations.

Typical weights: α = 1.0, β = 0.5, λ<sub>wm</sub> ≈ 0.7.  
The total training loss:

$$\mathcal{L}_{total}=\mathcal{L}_{plan}+\lambda_{wm}\mathcal{L}_{wm}$$

This encourages the planner's internal features to evolve smoothly in time, guided by realistic latent transitions. The weighting factor λ<sub>wm</sub> was determined through ablation studies to provide optimal balance between planning performance and world model accuracy.

---

## 📊 Preliminary Results on NuScenes-Mini Demo

| **Category** | **Metric** | **Enhanced (WM)** | **Vanilla SparseDrive** | **Delta** |
|:-------------:|:-----------|:-----------------:|:-----------------------:|:------:|
| **Detection** | mAP | 0.4021 | 0.4138 | –2.8 % |
|  | NDS | 0.4478 | 0.4512 | –0.8 % |
| **Tracking** | AMOTA | 0.5501 | 0.5253 | +4.7 % |
|  | AMOTP ↓ | 1.0319 | 1.0510 | +Quality ↑ |
|  | MOTA | 0.5495 | 0.5361 | +2.5 % |
|  | ID Switches ↓ | 64 | 108 | –40 % |
| **Mapping** | Boundary | 0.3940 | 0.3787 | +4.0 % |
| **Forecasting** | Ped EPA | 0.4447 | 0.3918 | +13.5 % |
|  | Car minADE ↓ | 0.4591 | 0.4682 | +1.9 % |
| **Planning** | Collision Rate ↓ | 1.127 % | 2.979 % | –62 % |
|  | L2 Error ↓ | 3.20 m | 3.74 m | –14 % |

**Initial Observations**
- ⚙️ **Temporal consistency:** Early results suggest potential improvements in tracking metrics (~5% AMOTA, ~40% fewer ID switches)  
- 🧭 **Planning performance:** Initial tests indicate possible reductions in collision rates and trajectory errors
- 🎯 **Motion forecasting:** Preliminary data shows promising trends for pedestrian and vehicle predictions

These early results suggest potential improvements across several metrics, though further experiments and rigorous statistical analysis are needed to confirm these findings. The initial reduction in collision rate is encouraging, but requires additional validation across more diverse scenarios before drawing definitive conclusions. Our experiments are ongoing, and we continue to refine both the baseline and enhanced models.  

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.6+ and cuDNN
- PyTorch 1.13.0
- mmcv_full 1.7.1
- mmdet 2.28.2
- numpy 1.23.5
- transformers 4.46.3 (for DINOv2 integration)
- flash-attn 2.3.2
- nuscenes-devkit 1.1.10

### Environment Setup

Follow these steps to set up the development environment:

```bash
# Clone this repository
git clone https://github.com/ahmeddawy/SparseDrive_WorldModel.git
cd SparseDrive_WorldModel

# Create a conda environment (recommended for isolation)
conda create -n sparsedrive_lwm python=3.8 -y
conda activate sparsedrive_lwm

# Set path and install PyTorch with CUDA 11.6 support
sparsedrive_path="$(pwd)"  # Get absolute path to repository
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt

# Compile the custom CUDA operators 
# This step is essential for the deformable attention mechanism
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../

# Verify installation by running a sanity check
python -c "import torch; import mmdet; import mmdet3d; print('Installation successful!')"
```

> **Note**: CUDA version 11.6 is required. If you have a different CUDA version, you'll need to adjust the PyTorch installation URLs accordingly.

### Data Preparation
1. Download the NuScenes dataset from [the official website](https://www.nuscenes.org/download)
2. Link or place the dataset in `data/nuscenes/`
3. Prepare the info files:
```bash
bash scripts/create_data.sh
```

4. Generate K-means clusters for anchors:
```bash
bash scripts/kmeans.sh
```

---

## 🚀 Usage

### Training
The training process is divided into two stages:

**Stage 1**: Perception pre-training (can be skipped if using provided checkpoint)
```bash
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1.py \
   1 \
   --deterministic
```

**Stage 2**: Joint training with Latent World Model
```bash
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2.py \
   1 \
   --deterministic
```

### Testing
Evaluate the model on the NuScenes mini dataset:
```bash
# Set the environment variable to use mini dataset
export NUSCENES_VERSION=mini

bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/sparsedrive_small_stage2/latest.pth \
    1 \
    --deterministic \
    --eval bbox
```

### Visualization
Visualize detection, tracking, mapping, and planning results:
```bash
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
    projects/configs/sparsedrive_small_stage2.py \
    --result-path work_dirs/sparsedrive_small_stage2/results.pkl
```

The visualization outputs will be saved in the `vis/` directory.