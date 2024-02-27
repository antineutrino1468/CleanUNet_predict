## Usage

1. **Prepare Noisy WAV Files:**
   Place all noisy `.wav` files that you want to clean in the `raw_wav` directory.

2. **Run the Script:**
   Execute `predict.py` using the following command:

```
  python predict.py
```

3. **Cleaned WAV Output:**
The cleaned `.wav` files will be generated in the `cleaned_wav` directory.

## Requirements

To synthesize [Microsoft DNS 2020](https://arxiv.org/ftp/arxiv/papers/2005/2005.13981.pdf) training data, you need [these dependencies](https://github.com/microsoft/DNS-Challenge/blob/interspeech2020/master/requirements.txt). If you just want to evaluate our pre-trained models on the test data, you may jump this.

Our code is tested on 8 NVIDIA V100 GPUs. You need to install very standard dependencies: ```numpy``` and ```scipy``` for scientific computing, ```torch, torchvision, torchaudio``` for deep learning and data loading, ```pesq, pystoi``` for audio evaluation, and ```tqdm``` for visualization.

## References

The code structure and distributed training are adapted from [WaveGlow (PyTorch)](https://github.com/NVIDIA/waveglow) (BSD-3-Clause license). The ```stft_loss.py``` is adapted from [ParallelWaveGAN (PyTorch)](https://github.com/kan-bayashi/ParallelWaveGAN) (MIT license). The self-attention blocks in ```network.py``` is adapted from [Attention is all you need (PyTorch)](https://github.com/jadore801120/attention-is-all-you-need-pytorch) (MIT license), which borrows from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (MIT license). The learning rate scheduler in ```util.py``` is adapted from [VQVAE2 (PyTorch)](https://github.com/rosinality/vq-vae-2-pytorch) (MIT license). Some utility functions are borrowed from [DiffWave (PyTorch)](https://github.com/philsyn/DiffWave-Vocoder) (MIT license) and [WaveGlow (PyTorch)](https://github.com/NVIDIA/waveglow) (BSD-3-Clause license).

For more evaluation methods, we refer readers to look at [SEGAN (PyTorch)](https://github.com/santi-pdp/segan_pytorch/blob/master/segan/utils.py) (MIT license). For more data augmentation methods, we refer readers to look at [FAIR-denoiser](https://github.com/facebookresearch/denoiser/blob/main/denoiser/augment.py) (CC-BY-NC 4.0 license). 

## Citation

```
@inproceedings{kong2022speech,
  title={Speech Denoising in the Waveform Domain with Self-Attention},
  author={Kong, Zhifeng and Ping, Wei and Dantrey, Ambrish and Catanzaro, Bryan},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7867--7871},
  year={2022},
  organization={IEEE}
}
```
