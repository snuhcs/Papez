## PAPEZ: RESOURCE-EFFICIENT SPEECH SEPARATION WITH AUDITORY WORKING MEMORY (ICASSP 2023)

Codes for ICASSP 2023 submission.

### Usage

1. Install dependencies through 

```
$ pip install -r requirements_pip.txt
```

2. Select desired configuration from the config directory and import them in `main.py`, `train.py` and 'test.py'.

3. Train a Papez model with:

```
$ python train.py --gpu $GPU_NUMBER
```

4. Test the model by setting the `ckpt_path` property of the config with the trained checkpoint path, and use the command

```bash
$ python test.py --gpu $GPU_NUMBER
```