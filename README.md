## IKC: Blind Super-Resolution With Iterative Kernel Correction
[ArXiv](https://arxiv.org/abs/1904.03377) | [BibTex](#citation)

Here is the code of ['Blind Super-Resolution With Iterative Kernel Correction'](https://www.jasongt.com/projectpages/IKC.html).<br/>
Based on [[BasicSR]](https://github.com/xinntao/BasicSR), [[MMSR]](https://github.com/open-mmlab/mmsr).

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
git clone https://github.com/knazeri/edge-connect.git
cd edge-connect
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
We use codes/scripts/generate_mod_LR_bic.py to produce LR/HR/Bicubic dataset

### Train

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
