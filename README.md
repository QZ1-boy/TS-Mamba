# TS-Mamba

The code of the paper "Trajectory-aware State Space Model for Online Video Super-Resolution".

# Requirements

CUDA==11.6 Python==3.7 Pytorch==1.13

## Environment
```python
conda create -n TSMamba python=3.7 -y && conda activate TSMamba

git clone --depth=1 https://github.com/QZ1-boy/TS-Mamba && cd QZ1-boy/TS-Mamba/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip3 install requirement.txt.

```

## Dataset Download
Training Datasets:

[REDS dataset] [REDS](https://seungjunnah.github.io/Datasets/reds.html),
[Vimeo-90K dataset] [Vimeo-90K](https://github.com/anchen1011/toflow)

Testing Datasets (Ground-Truth):

[REDS4 dataset] [REDS4](https://seungjunnah.github.io/Datasets/reds.html),
[Vid4 dataset] [Vid4](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)


# Train
```python
python3 basicsr/train.py -opt options/train/TSA/train_TSA.yaml.
```

# Test
```python
python3 basicsr/test.py -opt options/test/TSA/test_TSA.yaml 
```

# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2025fcvsr,
  title={FCVSR: A Frequency-aware Method for Compressed Video Super-Resolution},
  author={Zhu, Qiang and Zhang, Fan and Chen, Feiyu and Zhu, Shuyuan and Bull, David and Zeng, Bing},
  journal={arXiv preprint arXiv:2502.06431},
  year={2025}
}
```
# Related Work
Our work is built on TMP(https://github.com/xtudbxk/TMP) work, we also some online video super-resolution works, i.e., FDAN(xxx), KSNet(xxx). 

