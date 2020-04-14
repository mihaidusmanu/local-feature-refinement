import argparse

import cv2

import numpy as np

import os

import subprocess

from tqdm import tqdm


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='SURF feature extraction script')

    parser.add_argument(
        '--image_path', type=str, required=True,
        help='path to the images'
    )

    parser.add_argument(
        '--max_edge', type=int, default=1600,
        help='maximum image size at octave 0'
    )

    parser.add_argument(
        '--output_extension', type=str, default='.surf',
        help='extension for the output'
    )

    args = parser.parse_args()

    print(args)

    # Define SURF.
    surf = cv2.xfeatures2d.SURF_create()
    surf.setExtended(True)
    surf.setHessianThreshold(500)

    # Extract features.
    for image_name in os.listdir(args.image_path):
        image = cv2.imread(os.path.join(args.image_path, image_name))
        if image is None:
            continue
        image_size = image.shape

        # Resize such that max edge is 1600.
        image = image[:, :, ::-1]  # BGR -> RGB
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=-1)
        downscaling_factor = max(1, max(image.shape) / args.max_edge)
        image = cv2.resize(image, None, fx=(1 / downscaling_factor), fy=(1 / downscaling_factor), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract SURF.
        opencv_keypoints, opencv_descriptors = surf.detectAndCompute(gray_image, None)

        # Convert to numpy.
        keypoints = []
        scores = []
        for keypoint in opencv_keypoints:
            keypoints.append([keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle])
            scores.append(keypoint.response)
        keypoints = np.array(keypoints)
        if keypoints.shape[0] == 0:
            keypoints = np.zeros([0, 4])
        keypoints[:, : 2] *= downscaling_factor
        scores = np.array(scores)
        descriptors = opencv_descriptors
        if keypoints.shape[0] == 0:
            descriptors = np.zeros([0, 128])
        
        print('[%s] %dx%d, downscaling factor %.4f; %d keypoints' % (
            image_name, image_size[0], image_size[1], downscaling_factor, keypoints.shape[0]
        ))

        with open(os.path.join(args.image_path, image_name) + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                scores=scores,
                descriptors=descriptors
            )
