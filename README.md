# Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation

This is the PyTorch implementation of paper: **[Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation (LREC-COLING 2024)](<https://arxiv.org/abs/2403.01479>)**. 

We carry out our experiments on standard Transformer with the  [fairseq](https://github.com/pytorch/fairseq) toolkit. If you use any source code included in this repo in your work, please cite the following paper.

```bibtex
@misc{jin2024aligntodistill,
      title={Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation}, 
      author={Heegon Jin and Seonil Son and Jemin Park and Youngseok Kim and Hyungjong Noh and Yeonsoo Lee},
      year={2024},
      eprint={2403.01479},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone this_repository
cd fairseq
pip install --editable ./
```

We require a few additional Python dependencies:

``` bash
pip install sacremoses einops
```

# Prepare dataset

### IWSLT'14 German to English

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

First download and preprocess the data:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
# Training
First, you need train a teacher model, the training script is the same with fairseq. 
Second, use the trained teacher model to train an A2D student model. 
The '--teacher-ckpt-path' argument is used to specify the path to the trained teacher model checkpoint from the first step.

Adjustable arguments for experiments:
- add '--alpha' (default=0.5) : This argument controls the weight between the cross-entropy loss and the response-based distillation loss.
- add '--beta' (default=1) : This argument controls the weight between the response-based loss and the attention distillation loss.
- add '--decay' (default=0.9) : This argument sets the decay rate for the attention distillation loss.

Two scripts are provided for running the training processes:
- train_teacher.sh: This script is used to train the teacher model.
- train_student.sh: This script is used to train the A2D student model using the trained teacher model.

## Train a teacher model

```bash
bash train_teacher.sh
```

## Train a student model (with A2D method)

```bash
bash train_student.sh
```

## Test a student model (with A2D method)

```bash
bash test.sh
```

# Citation

Please cite as:

``` bibtex
@misc{jin2024aligntodistill,
      title={Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation}, 
      author={Heegon Jin and Seonil Son and Jemin Park and Youngseok Kim and Hyungjong Noh and Yeonsoo Lee},
      year={2024},
      eprint={2403.01479},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
