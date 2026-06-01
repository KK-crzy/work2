# Speckle2Self: Self-Supervised Ultrasound Speckle Reduction Without Clean Data

🔗 **Paper Links**  
- 📄 [ArXiv Version](https://arxiv.org/abs/2507.06828)  
- 🌐 [Project Page](https://noseefood.github.io/us-speckle2self/)  

PyTorch Implementation of the Paper [**Speckle2Self: Self-Supervised Ultrasound Speckle Reduction Without Clean Data**](https://arxiv.org/abs/2507.06828)

```
@article{li2025speckle2self,
  title={Speckle2Self: Self-supervised ultrasound speckle reduction without clean data},
  author={Li, Xuesong and Navab, Nassir and Jiang, Zhongliang},
  journal={Medical Image Analysis},
  pages={103755},
  year={2025},
  publisher={Elsevier}
}
```

<p align="center">
  <img src="demo/Sim_visual.png">
</p>

<p align="center">
  <img src="demo/CCA_visual.png">
</p>

# 0. Checklist

- [x] Inference Code :tada:
- [x] Training Code for Simulator dataset :tada:
- [x] Training Code for In-vivo dataset :tada:

# 1. Installation
Download **Speckle2Self Repo** with:
```
git clone https://github.com/noseefood/speckle2self-code.git
cd speckle2self-code
```
Our experiments are done with:

- Python 3.10.16
- PyTorch 2.6.0
- numpy 2.2.6
- opencv 4.11.0
- albumentations 2.0.3

# 2. Inference
Download pre-trained model and In-vivo or Simulation testset: [link](https://drive.google.com/drive/folders/1mIHPcwbXWxDtjKWxtpxfDqkJyP2li8ay?usp=sharing)
```
# In-vivo testset
python inference.py \
    --data_path data/inVivo/test_data.npy \
    --model_path model_2833.pth \
    --output_path results.npy \
    --visualize

# Simulation testset
python inference.py \
    --data_path data/simulator/test_data.npy \
    --model_path model_2999.pth \
    --output_path results.npy \
    --visualize
```

# 3. Training
Training on In-vivo or Simulation dataset:
```
python train.py
```

## ⚠️ Data Requirement for Best Performance

For optimal performance, please use **ultrasound envelope data** (not B-mode) with a minimum resolution of **512×512**.


