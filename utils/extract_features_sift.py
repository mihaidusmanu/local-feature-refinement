import argparse

import cv2

import imagesize

import numpy as np

import os

import sqlite3

import subprocess


def recover_database_images_and_ids(database_path):
    # Connect to the database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[1]] = row[0]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='SIFT feature extraction script')

    parser.add_argument(
        '--image_path', type=str, required=True,
        help='path to the images'
    )

    parser.add_argument(
        '--max_edge', type=int, default=1600,
        help='maximum image size at octave 0'
    )

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )

    parser.add_argument(
        '--output_extension', type=str, default='.sift',
        help='extension for the output'
    )

    args = parser.parse_args()

    database_path = os.path.join(args.image_path, 'features-sift.db')

    FNULL = open(os.devnull, 'w')
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'feature_extractor',
        '--database_path', database_path,
        '--image_path', args.image_path,
        '--SiftExtraction.max_image_size', str(args.max_edge),
        '--SiftExtraction.max_num_features', str(1000000),
        '--SiftExtraction.first_octave', '0'
    ], stdout=FNULL)
    FNULL.close()

    images = recover_database_images_and_ids(database_path)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    kps = {}
    cursor.execute('SELECT image_id, rows, cols, data FROM keypoints')
    for row in cursor:
        image_id = row[0]
        image_relative_path = images[image_id]
        path = os.path.join(args.image_path, image_relative_path)
        image_size = imagesize.get(path)
        downscaling_factor = max(1, max(image_size) / args.max_edge)
        if row[1] == 0:
            keypoints = np.zeros([0, 6])
        else:
            keypoints = np.frombuffer(row[-1], dtype=np.float32).reshape(row[1], row[2])
        # x, y, a11, a12, a21, a22
        # In COLMAP, the upper left pixel has the coordinate (0.5, 0.5).
        coords = keypoints[:, : 2] - .5
        a11 = keypoints[:, 2]
        a12 = keypoints[:, 3]
        a21 = keypoints[:, 4]
        a22 = keypoints[:, 5]
        scale = (np.sqrt(a11 * a11 + a21 * a21) + np.sqrt(a12 * a12 + a22 * a22)) / 2. / 1.6 / downscaling_factor
        ori = np.arctan2(a21, a11)

        if row[1] == 0:
            min_scale = 0
        else:
            min_scale = scale.min()

        print('[%s] %dx%d, downscaling factor %.4f; %d keypoints, %.4f min keypoint scale' % (
            image_relative_path, image_size[0], image_size[1], downscaling_factor, row[1], min_scale
        ))

        kps[image_relative_path] = np.concatenate([
            coords, scale[:, np.newaxis], ori[:, np.newaxis]
        ], axis=1)

    descrs = {}
    cursor.execute('SELECT image_id, rows, cols, data FROM descriptors')
    for row in cursor:
        image_id = row[0]
        image_relative_path = images[image_id]
        if row[1] == 0:
            descriptors = np.zeros([0, 128])
        else:
            descriptors = np.frombuffer(row[-1], dtype=np.uint8).reshape(row[1], row[2])
        descriptors = descriptors / np.linalg.norm(descriptors, axis=1)[:, np.newaxis]
        descrs[image_relative_path] = descriptors

    cursor.close()
    connection.close()

    os.remove(database_path)

    for _, image_relative_path in images.items():
        path = os.path.join(args.image_path, image_relative_path)
        keypoints = kps[image_relative_path]
        scores = np.zeros(keypoints.shape[0])
        descriptors = descrs[image_relative_path]
        with open(path + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                scores=scores,
                descriptors=descriptors
            )
