# Large Vision-Language Model for Medical Applications

## Dataset structure
```
root
├── images
│   ├── train
│   └── test
├── annotations
│   ├── train
│   │   └── grounded_diseases_train.json
│   └── test
│       └── grounded_diseases_test.json
└── pretrained_checkpoint
    └── checkpoint_stage3.pth
```

You may load from our pretrained model checkpoints:

For `checkpoint_stage3.pth`, you can load from the pretrained model below:
| MiniGPT-v2 (after stage-3) |
|------------------------------|
|[Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing) |

## Installation
- Python == 3.10.13
```bash
git clone https://github.com/ngocson1042002/LVLM-Med.git
cd LVLM-Med
pip install -r requirements.txt
```

## Training
### BiomedCLIP - Llama2
1. **Set Paths for Training**
    - Set the training annotations path to `root/annotations/test/grounded_diseases_train.json` [here](medlvlm/configs/datasets/vindrcxr/default.yaml#L6) at Line 6.

    - Set the training image path to `root/images/train` [here](medlvlm/configs/datasets/vindrcxr/default.yaml#L5) at Line 5.

    - Set the pretrained checkpoint path to `root/pretrained_checkpoint/checkpoint_stage3.pth` [here](train_configs/biomedclip_llama.yaml#L9) at Line 9.

    - Set the checkpoint save path [here](train_configs/biomedclip_llama.yaml#L44) at Line 44.

2. **Set Paths for Evaluation (After Training)**
    - Set the evaluation annotations path to `root/annotations/test/grounded_diseases_test.json` [here](eval_configs/eval_biomedclip_llama.yaml#L27) at Line 27.
    - Set the evaluation image path to `root/images/test` [here](eval_configs/eval_biomedclip_llama.yaml#L28) at Line 28.
    - Set the evaluation result output path [here](eval_configs/eval_biomedclip_llama.yaml#L38) at Line 38.

3. **Run**
```bash
torchrun --nproc-per-node NUM_GPU train.py\ 
         --cfg-path train_configs/biomedclip_llama.yaml\
         --cfg-eval-path eval_configs/eval_biomedclip_llama.yaml\
         --eval-dataset vindrcxr_val
```