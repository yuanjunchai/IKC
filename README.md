## IKC

Here is the code of ['Blind Super-Resolution With Iterative Kernel Correction'](https://www.jasongt.com/projectpages/IKC.html).<br/>
Based on [[BasicSR]](https://github.com/xinntao/BasicSR).

### Dataset Preparation
use script to produce LR dataset and corresponding kernel map

### Train

### Test

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

## Contributing

## Citation
    @InProceedings{gu2019blind,
        author = {Gu, Jinjin and Lu, Hannan and Zuo, Wangmeng and Dong, Chao},
        title = {Blind super-resolution with iterative kernel correction},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }
