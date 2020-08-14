import argparse

import os

import sqlite3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', required=True, type=str,
        help='path to the dataset'
    )
    
    args = parser.parse_args()

    database_path = os.path.join(args.dataset_path, 'database.db')

    # Recover images from database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    images = []
    cursor.execute('SELECT name FROM images;')
    for row in cursor:
        images.append(row[0])

    cursor.close()
    connection.close()

    # Save the exhaustive list of image pairs to match.
    f = open(os.path.join(args.dataset_path, 'image-list.txt'), 'w')

    for image_idx1 in range(len(images)):
        f.write('%s\n' % os.path.realpath(os.path.join(args.dataset_path, 'images', images[image_idx1])))

    f.close()
