# Instant-NeRF-Stylization
- Instant stylization between NeRF and NeRF, NeRF and image.
- Our code is based on the [instant-ngp](https://github.com/NVlabs/instant-ngp) implemented by [JNeRF](https://github.com/Jittor/JNeRF).

## Install(from [JNeRF](https://github.com/Jittor/JNeRF))
JNeRF environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))
* GPU supporting:
  * sm arch >= sm_61 (GTX 10x0 / TITAN Xp and above)
  * to use fp16: sm arch >= sm_70 (TITAN V / V100 and above). JNeRF will automatically use original fp32 if the requirements are not meet.
  * to use FullyFusedMLP: sm arch >= sm_75 (RTX 20x0 and above). JNeRF will automatically use original MLPs if the requirements are not meet.

**Step 1: Install the requirements**
```shell
sudo apt-get install tcl-dev tk-dev python3-tk
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JNeRF**

JNeRF is a benchmark toolkit and can be updated frequently, so installing in editable mode is recommended.
Thus any modifications made to JNeRF will take effect without reinstallation.

```shell
cd python
python -m pip install -e .
```

After installation, you can ```import jnerf``` in python interpreter to check if it is successful or not.



## Getting Started
### Datasets
- We use dataset nerf_synthetic and nerf_llff_data from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), style images from [AdaIN-style](https://github.com/xunhuang1995/AdaIN-style/tree/master/input/style).

### Config
- NeRF and NeRF.  "./projects/ngp_stylization/ngp_pair_base.py"
- NeRF and image. "./projects/ngp_stylization/ngp_pair_base_img.py"   For the image style target, there shall be a synthetic data for poses, details in the config. 

### Train
```shell
python tools/run_net.py --config-file ./projects/ngp_stylization/ngp_pair_base.py
```
### Test
```shell
python tools/run_net_stylization.py --config-file ./projects/ngp_stylization/ngp_pair_base.py --task test
```

## BibTeX

```
@misc{li2023instant,
      title={Instant Neural Radiance Fields Stylization}, 
      author={Shaoxu Li and Ye Pan},
      year={2023},
      eprint={2303.16884},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Related Work

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
