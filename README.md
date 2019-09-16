## IKC: Blind Super-Resolution With Iterative Kernel Correction
[ArXiv](https://arxiv.org/abs/1904.03377) | [BibTex](#citation)

Here is the code of ['Blind Super-Resolution With Iterative Kernel Correction'](https://www.jasongt.com/projectpages/IKC.html).<br/>
Based on [[BasicSR]](https://github.com/xinntao/BasicSR), [[MMSR]](https://github.com/open-mmlab/mmsr).
About more details, check [BasicSR](https://github.com/xinntao/BasicSR/tree/master/codes).

## Dependencies
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

## Installation
- Clone this repo:
```bash
git clone https://github.com/yuanjunchai/IKC.git
cd IKC
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip), [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) datasets. To train a model on the full dataset(DIV2K+Flickr2K, totally 3450 images), download datasets from official websites. 
After downloading, run [`codes/scripts/generate_mod_LR_bic.py`](codes/scripts/generate_mod_LR_bic.py) to generate LR/HR/Bicubic of train, test and validation datasets paths. 
```bash
python codes/scripts/generate_mod_LR_bic.py
```
## Getting Started
TODO: Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

### Train
First, train SFTMD network, and then use pretrained SFTMD to train Predictor and Corrector networks iteratively.

To train the SFTMD model, change image path of [`codes/options/train/train_SFTMD.yml`](codes/options/train/train_SFTMD.yml), especially dataroot_GT, dataroot_LQ. You could change opt['name'] to save different checkpoint filenames, and change opt['gpu_ids'] to assign specific GPU.
```bash
python codes/train_SFTMD.py -opt_F codes/options/train/train_SFTMD.yml
```
To train Predictor and Corrector models, you first should change opt_F['sftmd']['path']['pretrain_model_G']

### Test


## Contributing

## Citation
    @InProceedings{gu2019blind,
        author = {Gu, Jinjin and Lu, Hannan and Zuo, Wangmeng and Dong, Chao},
        title = {Blind super-resolution with iterative kernel correction},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }
