import os
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite

from dataset import load_CleanNoisyPairDataset
from util import print_size, sampling
from network import CleanUNet


def ezdenoise():
    """
    Denoise audio

    Parameters:
    output_directory (str):         save generated speeches to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    subset (str):                   training, testing, validation
    dump (bool):                    whether save enhanced (denoised) audio
    """
    with open('./configs/DNS-large-high.json') as f:
          data = f.read()

    config = json.loads(data)
    global gen_config
    gen_config              = config["gen_config"]
    global network_config
    network_config          = config["network_config"]      # to define wavenet
    global train_config
    train_config            = config["train_config"]        # train config
    global trainset_config
    trainset_config         = config["trainset_config"]     # to read trainset configurations

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    subset="predicting"
    ckpt_iter='pretrained'

    # setup local experiment path
    exp_path = train_config["exp_path"]
    # load data
    loader_config = deepcopy(trainset_config)
    
    loader_config["crop_length_sec"] = 0
    dataloader = load_CleanNoisyPairDataset(
        **loader_config, 
        subset=subset,
        batch_size=1, 
        num_gpus=1
    )

    # predefine model
    net = CleanUNet(**network_config).cuda()
    print_size(net)

    # load checkpoint
    ckpt_directory = os.path.join(train_config["log"]["directory"], exp_path, 'checkpoint')
    model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # inference
    sortkey = lambda name: '_'.join(name.split('/')[-1].split('_')[1:])
    for _, noisy_audio, fileid in tqdm(dataloader):
        filename = sortkey(fileid[1][0])

        noisy_audio = noisy_audio.cuda()
        LENGTH = len(noisy_audio[0].squeeze())
        generated_audio = sampling(net, noisy_audio)
        
        
        tks=filename.split("\\")

        if not os.path.exists('./cleaned_wav/'):
            os.mkdir('./cleaned_wav/')
        
        wavwrite(os.path.join(f'./cleaned_wav/', 'enhanced_{}'.format(tks[-1])), 
                trainset_config["sample_rate"],
                generated_audio[0].squeeze().cpu().numpy())

if __name__ == "__main__":
    ezdenoise()
