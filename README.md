# TS-Mamba

The code of the paper "Trajectory-aware Shifted State Space Models for Online Video Super-Resolution".

# Requirements

CUDA==11.7 Python==3.9 Pytorch==1.13.1

## Environment
```python
conda create -n TSMamba python=3.9 -y && conda activate TSMamba

git clone --depth=1 https://github.com/QZ1-boy/TS-Mamba && cd QZ1-boy/TS-Mamba/

python -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio--0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirement.txt
pip install causal_conv1d==1.0.0
pip install mamba-ssm==1.0.1
pip install open-python timm numpy tqdm scipy tensorboard transformers==4.33.0 spatial-correlation-sampler==0.3.0
cd . /basicsr/ops/msda
bash make.sh
python test.py
```

## Dataset Download
Training Datasets:

REDS dataset: [here](https://seungjunnah.github.io/Datasets/reds.html),
Vimeo-90K dataset: [here](https://github.com/anchen1011/toflow).

Testing Datasets:

REDS4: [here](https://seungjunnah.github.io/Datasets/reds.html),
Vid4 dataset: [here](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA).


# Train
```python
python3 basicsr/train.py -opt options/train/TSA/train_TSA.yaml
```

# Test
```python
python3 basicsr/test.py -opt options/test/TSA/test_TSA.yaml 
```

# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2025trajectory,
  title={Trajectory-aware Shifted State Space Models for Online Video Super-Resolution},
  author={Zhu, Qiang and Meng, Xiandong and Jiang, Yuxian and Zhang, Fan and Bull, David and Zhu, Shuyuan and Zeng, Bing},
  journal={arXiv preprint arXiv:2508.10453},
  year={2025}
}
```
# Related Work
Our work is built on [TMP](https://github.com/xtudbxk/TMP), we also release some online video super-resolution works, i.e., [FDAN](https://github.com/IanYeung/EfficientVSR), [KSNet](xxx). 

