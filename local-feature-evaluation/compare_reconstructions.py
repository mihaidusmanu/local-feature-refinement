import argparse

import numpy as np

import os

import shutil

import types

import sys

import subprocess


def recover_images(scene_path):
    images = {}
    with open(os.path.join(scene_path, 'images.txt')) as images_file:
        lines = images_file.readlines()
        lines = lines[4 :]  # Skip the header.
        raw_poses = [line.strip('\n').split(' ') for line in lines[:: 2]]
        for raw_pose in raw_poses:
            image_id = int(raw_pose[0])
            image_name = raw_pose[-1]
            images[image_name] = image_id
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to COLMAP executable folder'
    )

    parser.add_argument(
        '--raw_reconstruction', type=str, required=True,
        help='path to the reconstruction using raw keypoints (without refinement)'
    )
    parser.add_argument(
        '--ref_reconstruction', type=str, required=True,
        help='path to the reconstruction using refined keypoints'
    )

    args = parser.parse_args()

    raw_images = recover_images(args.raw_reconstruction)
    ref_images = recover_images(args.ref_reconstruction)

    raw_extra_images = list(set(raw_images.keys()) - set(ref_images.keys()))
    ref_extra_images = list(set(ref_images.keys()) - set(raw_images.keys()))

    # Prepare list for image_deleter.
    with open(os.path.join(args.raw_reconstruction, 'extra_ids.txt'), 'w') as f:
        for image_name in raw_extra_images:
            f.write('%d\n' % raw_images[image_name])

    # Delete the image from the model.
    if not os.path.isdir(os.path.join(args.raw_reconstruction, 'common')):
        os.mkdir(os.path.join(args.raw_reconstruction, 'common'))
    
    FNULL = open(os.devnull, 'w')
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'image_deleter',
        '--input_path', args.raw_reconstruction,
        '--output_path', os.path.join(args.raw_reconstruction, 'common'),
        '--image_ids_path', os.path.join(args.raw_reconstruction, 'extra_ids.txt')
    ], stdout=FNULL)
    FNULL.close()

    # Print statistics to stdout.
    print('======================')
    print('Raw reconstruction')
    print('======================')
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'model_analyzer',
        '--path', os.path.join(args.raw_reconstruction, 'common')
    ])
    print()

    # Prepare list for image_deleter.
    with open(os.path.join(args.ref_reconstruction, 'extra_ids.txt'), 'w') as f:
        for image_name in ref_extra_images:
            f.write('%d\n' % ref_images[image_name])

    # Delete the image from the model.
    if not os.path.isdir(os.path.join(args.ref_reconstruction, 'common')):
        os.mkdir(os.path.join(args.ref_reconstruction, 'common'))
    
    FNULL = open(os.devnull, 'w')
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'image_deleter',
        '--input_path', args.ref_reconstruction,
        '--output_path', os.path.join(args.ref_reconstruction, 'common'),
        '--image_ids_path', os.path.join(args.ref_reconstruction, 'extra_ids.txt')
    ], stdout=FNULL)
    FNULL.close()

    # Print statistics to stdout.
    print('======================')
    print('Refined reconstruction')
    print('======================')
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'model_analyzer',
        '--path', os.path.join(args.ref_reconstruction, 'common')
    ])
