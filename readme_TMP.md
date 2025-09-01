## HOW TO USE

- ##### Prerequisite
  We train and test our project under torch==1.10 and python3.7. You can install the required libs with `pip3 install requirement.txt`.

- ##### Dataset
  Please refer to [here](https://github.com/xinntao/EDVR/blob/master/docs/Datasets.md) to download the *REDS*, *Vimeo90K*  dataset and [there](https://mmagic.readthedocs.io/en/stable/dataset_zoo/vid4.html) for*Vid4* dataset.

- ##### Train
  You can train this project using `CUDA_VISIBLE_DEVICES=1 python3 basicsr/train.py -opt options/train/TMP/train_TMP.yaml`.

- ##### Test
  You can test the trained models using `python3 basicsr/test.py -opt options/test/TMP/test_TMP.yaml`.

- ##### Pretrained Models
  Please download the pretrained models from [OneDrive](https://connectpolyu-my.sharepoint.com/:u:/g/personal/22040257r_connect_polyu_hk/Eabm8iorllFBpZm7JU4DED0BFgDUqcA8IUBJ_nYfh62G2A?e=7JnYap).

*please modify the paths of dataset and the trained model in the corresponding config file mannually*


FastonlineVSR: https://github.com/IanYeung/EfficientVSR


Install environment:

conda create -n tmp python==3.9
conda activate tmp
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install causal_conv1d==1.0.0
pip install mamba-ssm==1.0.1
pip install opencv-python timm numpy tqdm scipy tensorboard
pip install transformers==4.33.0
pip install spatial-correlation-sampler==0.3.0
cd ./basicsr/ops/msda
bash make.sh
python test.py