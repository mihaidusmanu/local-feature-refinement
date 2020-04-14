import argparse

import numpy as np

import os

import subprocess

import sqlite3

import types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to the dataset'
    )

    args = parser.parse_args()

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(
        args.dataset_path, 'database.db'
    )
    paths.image_path = os.path.join(
        args.dataset_path, 'images'
    )

    # Run feature extraction to create the database with images and cameras.
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'feature_extractor',
        '--database_path', paths.database_path, 
        '--image_path', paths.image_path,
        '--SiftExtraction.max_image_size', '50'
    ])

    # Empty the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    cursor.execute('DELETE FROM keypoints;')
    cursor.execute('DELETE FROM descriptors;')
    cursor.execute('DELETE FROM matches;')
    cursor.execute('DELETE FROM two_view_geometries;')

    connection.commit()

    # Close the connection to the database.
    cursor.close()
    connection.close()
