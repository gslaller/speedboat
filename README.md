# Speedboat

GPU Transformer training script, a derivative of [KellerJordan modded-nanogpt's](https://github.com/KellerJordan/modded-nanogpt) repo.  
This repo provides a lean implementation running on a single GPU(like GH200, priced at 1.49$/hr). It's a great base script for trying out new transformer based language modeling ideas.  
There is a run of a slightly older version [here](/runs/1744157764_6158e650.txt).

## Some diffs keller's script:

1. removal of kernel & mask length warmup.
2. simpler long, short masking for flex attention.
3. remove of FP8 for the Linear Layers.

## Installation

```bash
pip install -r requirements.txt
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

cd data && python download.py && cd ..
python train.py
```

## TODOs

- [ ] Clean up code and provide nice commenting
- [ ] Add hellaswag benchmark, for eval
- [ ] Add 350M param model
