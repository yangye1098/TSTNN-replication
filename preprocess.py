import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
import os

import torchaudio


def main(dir, out, sample_rate):
    resampler = None
    filenames = glob(f'{dir}/**/*.wav', recursive=True)
    for i, filename in tqdm(enumerate(filenames), desc='Preprocessing', total=len(filenames)):
        sound_original, sr = torchaudio.load(filename)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(sr, sample_rate, dtype=sound_original.dtype)
            sr_original = sr
        else:
            assert sr_original == sr, f'The sample rate of all files must be the same, {filename} is {sr}Hz'
        sound = resampler(sound_original)
        target_filename = filename.replace(dir, out)
        target_path, _ = os.path.split(target_filename)
        target_path = Path(target_path)
        if not target_path.exists():
            Path(target_path).mkdir(parents=True, exist_ok=True)
        torchaudio.save(target_filename, sound, sample_rate)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Preprocess data')
    args.add_argument('dir', type=str,
                        help='the directory containing subfolders which contain wav files to be prepared')
    args.add_argument('out', type=str,
                      help='the directory to put the processed files')
    args.add_argument('sample_rate', type=int,
                      help='the sample rate to resample')
    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    main(args.dir, args.out, args.sample_rate)
