import os
from tqdm import tqdm
import argparse
import subprocess

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='SIFT feature extraction script')

    parser.add_argument(
        '--directory_path', type=str, required=True,
        help='path to the directory'
    )
    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )
    args = parser.parse_args()

    for seq in tqdm(os.listdir(args.directory_path)):
        seq_path = os.path.join(args.directory_path, seq)
        subprocess.call([
            'python', 'utils/extract_features_sift.py',
            '--image_path', seq_path,
            '--colmap_path', args.colmap_path
        ])