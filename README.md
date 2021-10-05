# Codebase for Multi-Label Iterated Learning (MILe)
Welcome to the codebase for the ICLR2022 submission titled "Overcoming Label Ambiguity with Multi-Label Iterated Learning"

## Setup
We assume the user to have a data directory which is stored in the `bash` variable `$DATADIR`. `$DATADIR` should contain a folder with the ILSVRC2012 dataset. The folder should contain `imagenet/train` and `imagenet/val`.
1. `pip install -r requirements.txt`
2. Download ImageNet ReaL labels:
```bash
wget https://raw.githubusercontent.com/google-research/reassessed-imagenet/master/real.json
mkdir -p ./$DATADIR/real
mv real.json ./$DATADIR/real/
```

Usage:
```bash
python3 trainval_iters.py -e iterative_original_imagenet_real_k1k2 -sb logs -d $DATADIR
```
You can adjust the number of teacher (k1) and student (k2) iterations in `exp_configs.py`.

