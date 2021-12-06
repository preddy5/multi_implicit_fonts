# A Multi-Implicit Neural Representation for Fonts (NeurIPS 2021)
[Project Page](http://geometry.cs.ucl.ac.uk/projects/2021/multi_implicit_fonts/) | [Paper](https://arxiv.org/abs/2106.06866)

<img src="http://geometry.cs.ucl.ac.uk/projects/2021/multi_implicit_fonts/paper_docs/teaser.png">

**Abstract**:
Fonts are ubiquitous across documents and come in a variety of styles. They are either represented in a native vector format or rasterized to produce fixed resolution images. In the first case, the non-standard representation prevents benefiting from latest network architectures for neural representations; while, in the latter case, the rasterized representation, when encoded via networks, results in loss of data fidelity, as font-specific discontinuities like edges and corners are difficult to represent using neural networks. Based on the observation that complex fonts can be represented by a superposition of a set of simpler occupancy functions, we introduce multi-implicits to represent fonts as a permutation-invariant set of learned implict functions, without losing features (e.g., edges and corners). However, while multi-implicits locally preserve font features, obtaining supervision in the form of ground truth multi-channel signals is a problem in itself. Instead, we propose how to train such a representation with only local supervision, while the proposed neural architecture directly finds globally consistent multi-implicits for font families. We extensively evaluate the proposed representation for various tasks including reconstruction, interpolation, and synthesis to demonstrate clear advantages with existing alternatives. Additionally, the representation naturally enables glyph completion, wherein a single characteristic font is used to synthesize a whole font family in the target style.

# Data
Download data from the links below and extract them in the 'data/' folder.

You can find example data without corners here:

`https://geometry.cs.ucl.ac.uk/projects/2021/multi_implicit_fonts/paper_docs/renders.zip`

You can find example data with corners here:

`https://geometry.cs.ucl.ac.uk/projects/2021/multi_implicit_fonts/paper_docs/renders_3c.zip`

*Note the project is done as a part of a research internship and I do not have access to the data anymore, so you'll have to generate the data yourself for future experiment. I apologize for the inconvience.*

# Training

Train using this command in the root folder.

Run the following command for training without local supervsion:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config family_wo_local --version 42
```
Run the below command to train with local supervsion:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config family_w_local --version 43 --local_supervision
```

# Inference
Use local_eval python notebook in the logs folder to generate beautiful results.


## Citation
```
@article{reddy2021multi,
title={A Multi-Implicit Neural Representation for Fonts},
author={Reddy, Pradyumna and Zhang, Zhifei and Fisher, Matthew and Jin, Hailin and Wang, Zhaowen and Mitra, Niloy J},
journal={arXiv preprint arXiv:2106.06866},
year={2021}
}
			
```
