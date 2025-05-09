from tqdm import tqdm
import librosa
from librosa.core import load
from librosa.util import normalize
import torch
import torch.nn.functional as F
from utils import *
import argparse
from pathlib import Path
import time
from networks import UNetFilter
from torch.autograd import Variable
import glob
from modules import MelGAN_Generator, Audio2Mel
from pathlib import Path
import random
import pdb
import math


LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--device", type = str, default = '0')
    parser.add_argument("--filter_receptive_field", type = int, default = 3)
    parser.add_argument("--n_mel_channels", type = int, default = 80)
    parser.add_argument("--ngf", type = int, default = 32)
    parser.add_argument("--n_residual_layers", type = int, default=3)
    parser.add_argument("--sampling_rate", type = int, default=16000)
    parser.add_argument("--seeds", type=int, nargs='+', default=[123])
    parser.add_argument("--num_runs", type = int, default = 1)
    parser.add_argument("--noise_dim", type=int, default=65)
    parser.add_argument("--max_duration", type=int, default=16.7)
    parser.add_argument("--path_to_dir", type=str, default='C:\\Users\\C\\Documents\\Coding Projects\\ASR\\test_dutch\\test\\audio')
    parser.add_argument("--path_to_models", type=str, default='C:\\Users\\C\\Documents\\Coding Projects\\ASR\\GenGAN\\models')
    args = parser.parse_args()
    return args


def load_wav_to_torch(full_path, max_duration):
   audio, sampling_rate = load(full_path, sr=16000)
   audio = 0.95 * normalize(audio)
   duration = len(audio)
   audio = torch.from_numpy(audio).float()
   # utterances of segment_length
   if audio.size(0) <= max_duration*sampling_rate:
       audio = F.pad(audio, (0, int(max_duration*sampling_rate) - audio.size(0)), "constant").data
   return audio, duration


def main():
    args = parse_args()
    root = Path(os.getcwd())
    device = 'cuda:' + str(args.device)

    set_seed(args.seeds[0])  # Only use the first seed if multiple given
    audio_dir = Path(args.path_to_dir)
    run_dir = root / 'audio_outputs_dutch'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    fft = Audio2Mel(sampling_rate=args.sampling_rate)
    Mel2Audio = MelGAN_Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    # Mel2Audio.load_state_dict(torch.load(Path(args.path_to_models) / 'multi_speaker.pt'))
    Mel2Audio.load_state_dict(torch.load('./data_files/multi_speaker.pt'))

    netG = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128],
                      kernel_size=args.filter_receptive_field,
                      image_width=32, image_height=80, noise_dim=args.noise_dim,
                      nb_classes=2, embedding_dim=16, use_cond=False).to(device)
    netG.load_state_dict(torch.load(Path(args.path_to_models) / 'netG_epoch_25.pt'))
    netG.eval()

    print(f"Reading from {audio_dir}")

    for wav_file in audio_dir.glob("**/*.flac"):
        print(f"Processing {wav_file.name}")
        x, dur = load_wav_to_torch(str(wav_file), args.max_duration)
        x = torch.unsqueeze(x, 1)
        spectrograms = fft(x.reshape(1, x.size(0))).detach()
        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
        spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

        z = torch.randn(spectrograms.shape[0], args.noise_dim * 5).to(device)
        gen_secret = Variable(LongTensor(np.random.choice([1.0], spectrograms.shape[0]))).to(device)
        y_n = gen_secret * np.random.normal(0.5, math.sqrt(0.05))
        generated_neutral = netG(spectrograms, z, y_n).detach()

        generated_neutral = torch.squeeze(generated_neutral, 1).to(device) * 3 * stds.to(device) + means.to(device)
        inverted_neutral = Mel2Audio(generated_neutral).squeeze().detach().cpu()

        output_path = run_dir / f"{wav_file.stem}_transformed.wav"
        save_sample(str(output_path), args.sampling_rate, inverted_neutral[:dur])

    print(f"All files processed. Output saved in: {run_dir}")


if __name__ == "__main__":
    main()
