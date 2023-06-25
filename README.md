## Official Implementation of `Papez: Resource-Efficient Speech Separation with Auditory Working Memory` (ICASSP 2023)

> [Hyunseok Oh](http://www.aistudy.co.kr/ohs/), [Juheon Yi](https://juheonyi.github.io/), [Youngki Lee](http://youngkilee.blogspot.com/)<br>
> In ICASSP 2023. <br>

> Paper: https://ieeexplore.ieee.org/document/10095136<br>
> Slides: https://drive.google.com/file/d/1uksC183JlXdGwQ83rJgu-VFBNfLlE_r0/view?usp=sharing<br>
> Poster: https://drive.google.com/file/d/1h6wLwyAfA_A8xODHVKLI6zREkefI2h3c/view?usp=sharing<br>
> Video: https://drive.google.com/file/d/1hANUv-7_0S40A1jrfdRJ0yyR-FgNrwnv/view?usp=sharing<br>

> **Abstract:** *Transformer-based models recently reached state-of-the-art single-channel speech separation accuracy; However, their extreme computational load makes it difficult to deploy them in resource-constrained mobile or IoT devices. We thus present Papez, a lightweight and computation-efficient single-channel speech separation model. Papez is based on three key techniques. We first replace the inter-chunk Transformer with small-sized auditory working memory. Second, we adaptively prune the input tokens that do not need further processing. Finally, we reduce the number of parameters through the recurrent transformer. Our extensive evaluation shows that Papez achieves the best resource and accuracy tradeoffs with a large margin.*

<p  align="middle">
  <img src="docs/images/plot.png" width="900" />
</p>

### Usage

1. Install the dependencies through 

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

## Citation
Please cite our paper if you find our work useful: 
```
@inproceedings{oh2023papez,
  title={Papez: Resource-Efficient Speech Separation with Auditory Working Memory},
  author={Oh, Hyunseok and Yi, Juheon and Lee, Youngki},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

