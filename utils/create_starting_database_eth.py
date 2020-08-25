import argparse

import numpy as np

import os

import shutil

import subprocess

import sqlite3

import types

import sys


def array_to_blob(array):
    return array.tostring()


def parse_empty_reconstruction(dummy_database_path, reference_model_path):
    # Connect to the database.
    connection = sqlite3.connect(dummy_database_path)
    cursor = connection.cursor()

    cursor.execute('DELETE FROM cameras;')
    cursor.execute('DELETE FROM images;')
    cursor.execute('DELETE FROM keypoints;')
    cursor.execute('DELETE FROM descriptors;')
    cursor.execute('DELETE FROM matches;')
    cursor.execute('DELETE FROM two_view_geometries;')

    with open(os.path.join(reference_model_path, 'cameras.txt')) as cameras_file:
        lines = cameras_file.readlines()
        lines = lines[3 :]  # Skip the header.
        for raw_camera_info in lines:
            raw_camera_info = raw_camera_info.strip('\n').split(' ')
            camera_id = int(raw_camera_info[0])
            camera_params = np.array(list(map(float, raw_camera_info[2 :])))
            if raw_camera_info[1] == 'PINHOLE':
                cursor.execute(
                    'INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);',
                    (camera_id, 1, camera_params[0], camera_params[1], array_to_blob(camera_params[2 :].astype(np.float64)), 1)
                )
            else:
                cursor.execute(
                    'INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);',
                    (camera_id, 0, camera_params[0], camera_params[1], array_to_blob(camera_params[2 :].astype(np.float64)), 1)
                )

    with open(os.path.join(reference_model_path, 'images.txt')) as images_file:
        lines = images_file.readlines()
        lines = lines[4 :]  # Skip the header.
        for raw_image_info in lines[:: 2]:
            raw_image_info = raw_image_info.strip('\n').split(' ')
            image_id = int(raw_image_info[0])
            camera_id = int(raw_image_info[-2])
            image_name = raw_image_info[-1]
            cursor.execute(
                'INSERT INTO images(image_id, name, camera_id) VALUES(?, ?, ?);',
                (image_id, image_name, camera_id)
            )

    connection.commit()

    # Close the connection to the database.
    cursor.close()
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )

    parser.add_argument(
        '--dataset_path', required=True,
        help='Path to the dataset'
    )

    args = parser.parse_args()

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.dummy_database_path = os.path.join(
        args.dataset_path, 'database.db'
    )
    paths.reference_model_path = os.path.join(
        args.dataset_path, 'dslr_calibration_undistorted'
    )

    # Create database.
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'database_creator',
        '--database_path', paths.dummy_database_path
    ])
    parse_empty_reconstruction(paths.dummy_database_path, paths.reference_model_path)
