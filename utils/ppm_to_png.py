import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', required=True, type=str,
        help='path to the dataset that contains a set of sequences with ppm images'
    )
    
    args = parser.parse_args()

    for seq in tqdm(os.listdir(args.dataset_path)):
        seq_path = os.path.join(args.dataset_path, seq)
        for ppm_name in [fname for fname in os.listdir(seq_path) if fname.endswith('.ppm')]:
            img = Image.open(os.path.join(seq_path, ppm_name))
            img.save(os.path.join(seq_path, ppm_name[:-4] + '.png'), format="png")
