import argparse

import os

import sqlite3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', required=True, type=str,
        help='path to the images'
    )
    parser.add_argument(
        '--num_frames', default=3, type=str,
        help='number of frames for sequential matching'
    )
    
    args = parser.parse_args()

    database_path =  os.path.join(args.dataset_path, 'database.db')

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
    f = open(os.path.join(args.dataset_path, 'match-list.txt'), 'w')

    for image_idx1 in range(len(images)):
        _, tail = os.path.split(images[image_idx1])
        frame_idx1 = int(tail.split('-')[0])
        for image_idx2 in range(image_idx1 + 1, len(images)):
            _, tail = os.path.split(images[image_idx2])
            frame_idx2 = int(tail.split('-')[0])
            if (frame_idx2 - frame_idx1) > args.num_frames:
                continue
            f.write('%s %s\n' % (images[image_idx1], images[image_idx2]))

    f.close()

