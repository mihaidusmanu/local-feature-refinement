# Adapted from https://github.com/ahojnnes/local-feature-evaluation/blob/master/scripts/reconstruction_pipeline.py - Copyright 2017, Johannes L. Schoenberger <jsch at inf.ethz.ch>

import argparse

import glob

import json

import multiprocessing

import numpy as np

import os

import shutil

import sqlite3

import subprocess

import types

import types_pb2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )

    parser.add_argument(
        '--method_name', type=str, required=True,
        help='name of the method'
    )
    parser.add_argument(
        '--matches_file', type=str, required=True,
        help='path to the matches file'
    )
    parser.add_argument(
        '--solution_file', type=str, default=None,
        help='path to the multi-view optimization solution file (leave None for no refinement)'
    )

    parser.add_argument(
        '--output_file', type=str, required=True,
        help='path to the output file'
    )

    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def array_to_blob(array):
    return array.tostring()


def complete_keypoints(keypoints):
    if keypoints.shape[1] == 2:
        return np.hstack([
            keypoints, np.ones([keypoints.shape[0], 1]), np.zeros([keypoints.shape[0], 1])
        ])
    elif keypoints.shape[1] == 3:
        return np.hstack([
            keypoints, np.zeros([keypoints.shape[0], 1])
        ])
    else:
        return keypoints


def import_matches(colmap_path, method_name, database_path, image_path, match_list_path, matches_file, solution_file):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute(
        'SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'inlier_matches\';'
    )
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute('DELETE FROM keypoints;')
    cursor.execute('DELETE FROM descriptors;')
    cursor.execute('DELETE FROM matches;')
    if inlier_matches_table_exists:
        cursor.execute('DELETE FROM inlier_matches;')
    else:
        cursor.execute('DELETE FROM two_view_geometries;')
    connection.commit()

    images = {}
    cursor.execute('SELECT name, image_id FROM images;')
    for row in cursor:
        images[row[0]] = row[1]

    # Import the features.
    if solution_file is not None:
        solution_file_proto = types_pb2.SolutionFile()
        with open(solution_file, 'rb') as f:
            solution_file_proto.ParseFromString(f.read())

        image_proto_idx = {}
        for idx, image in enumerate(solution_file_proto.images):
            image_proto_idx[image.image_name] = idx

    sum_num_features = 0
    for image_name, image_id in images.items():
        keypoint_path = os.path.join(image_path, '%s.%s' % (image_name, method_name))
        features = np.load(keypoint_path, allow_pickle=True)

        keypoints = features['keypoints'][:, : 3]
        if keypoints.shape[0] == 0:
            keypoints = np.zeros([0, 4])
        keypoints = complete_keypoints(keypoints).astype(np.float32)

        num_features = keypoints.shape[0]
        sum_num_features += num_features

        if solution_file is not None:
            displacements = np.zeros([num_features, 2]).astype(np.float32)
            if image_name in image_proto_idx:
                for displacement in solution_file_proto.images[image_proto_idx[image_name]].displacements:
                    feature_idx = displacement.feature_idx
                    di = displacement.di
                    dj = displacement.dj
                    displacements[feature_idx, :] = [dj, di]
                fact = solution_file_proto.images[image_proto_idx[image_name]].fact
                displacements *= fact
            keypoints[:, : 2] += displacements * 16
        keypoints[:, : 2] += 0.5

        # descriptors = features['descriptors']

        assert keypoints.shape[1] == 4
        # assert keypoints.shape[0] == descriptors.shape[0]
        keypoints_str = array_to_blob(keypoints)
        cursor.execute(
            'INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);',
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str)
        )
    connection.commit()

    matches_files = []
    if os.path.exists(matches_file):
        matches_files.append(matches_file)
    else:
        part_idx = 0
        while os.path.exists('%s.part.%d' % (matches_file, part_idx)):
            matches_files.append('%s.part.%d' % (matches_file, part_idx))
            part_idx += 1

    image_pairs = []
    image_pair_ids = set()

    for matches_file in matches_files:
        matching_file_proto = types_pb2.MatchingFile()
        with open(matches_file, 'rb') as f:
            matching_file_proto.ParseFromString(f.read())

        for image_pair in matching_file_proto.image_pairs:
            image_name1, image_name2 = image_pair.image_name1, image_pair.image_name2
            image_pairs.append((image_name1, image_name2))
            image_id1, image_id2 = images[image_name1], images[image_name2]
            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
            if image_pair_id in image_pair_ids:
                continue
            image_pair_ids.add(image_pair_id)

            matches = []
            for match in image_pair.matches:
                matches.append([int(match.feature_idx1), int(match.feature_idx2)])

            matches = np.array(matches).astype(np.uint32)
            if matches.shape[0] == 0:
                matches = np.zeros([0, 2])
            assert matches.shape[1] == 2
            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]
            matches_str = array_to_blob(matches)
            cursor.execute(
                'INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);',
                (image_pair_id, matches.shape[0], matches.shape[1], matches_str)
            )
        connection.commit()

    cursor.close()
    connection.close()

    subprocess.call([
        os.path.join(colmap_path, 'colmap'), 'matches_importer',
        '--database_path', database_path,
        '--match_list_path', match_list_path,
        '--match_type', 'pairs'
    ])

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute('SELECT count(*) FROM images;')
    num_images = next(cursor)[0]

    cursor.execute('SELECT count(*) FROM two_view_geometries WHERE rows > 0;')
    num_inlier_pairs = next(cursor)[0]

    cursor.execute('SELECT sum(rows) FROM two_view_geometries WHERE rows > 0;')
    num_inlier_matches = next(cursor)[0]

    cursor.close()
    connection.close()

    return dict(
        num_images=num_images,
        num_inlier_pairs=num_inlier_pairs,
        num_inlier_matches=num_inlier_matches,
        avg_num_features=(sum_num_features / num_images)
    )


def reconstruct(colmap_path, database_path, image_path, sparse_path):
    # Run the sparse reconstruction.
    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)
    subprocess.call([
        os.path.join(colmap_path, 'colmap'), 'mapper',
        '--database_path', database_path,
        '--image_path', image_path,
        '--output_path', sparse_path,
        '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 8))
    ])

    # Find the largest reconstructed sparse model.
    models = os.listdir(sparse_path)
    if len(models) == 0:
        print('Warning: Could not reconstruct any model')
        return

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        subprocess.call([
            os.path.join(colmap_path, 'colmap'), 'model_converter',
            '--input_path', os.path.join(sparse_path, model),
            '--output_path', os.path.join(sparse_path, model),
            '--output_type', 'TXT'
        ])
        with open(os.path.join(sparse_path, model, 'cameras.txt'), 'r') as fid:
            for line in fid:
                if line.startswith('# Number of cameras'):
                    num_images = int(line.split()[-1])
                    if num_images > largest_model_num_images:
                        largest_model = model
                        largest_model_num_images = num_images
                    break

    assert(largest_model_num_images > 0)

    largest_model_path = os.path.join(sparse_path, largest_model)

    # Recover model statistics.
    stats = subprocess.check_output([
        os.path.join(colmap_path, 'colmap'), 'model_analyzer',
        '--path', largest_model_path
    ])

    stats = stats.decode().split('\n')
    for stat in stats:
        if stat.startswith('Registered images'):
            num_reg_images = int(stat.split()[-1])
        elif stat.startswith('Points'):
            num_sparse_points = int(stat.split()[-1])
        elif stat.startswith('Observations'):
            num_observations = int(stat.split()[-1])
        elif stat.startswith('Mean track length'):
            mean_track_length = float(stat.split()[-1])
        elif stat.startswith('Mean observations per image'):
            num_observations_per_image = float(stat.split()[-1])
        elif stat.startswith('Mean reprojection error'):
            mean_reproj_error = float(stat.split()[-1][:-2])

    return dict(
        num_reg_images=num_reg_images,
        num_sparse_points=num_sparse_points,
        num_observations=num_observations,
        mean_track_length=mean_track_length,
        num_observations_per_image=num_observations_per_image,
        mean_reproj_error=mean_reproj_error
    )


def main():
    args = parse_args()
    refine = (args.solution_file is not None)

    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(args.dataset_path, '%s-%s.db' % (args.method_name, 'ref' if refine else 'raw'))
    paths.image_path = os.path.join(args.dataset_path, 'images')
    paths.match_list_path = os.path.join(args.dataset_path, 'match-list.txt')
    paths.sparse_path = os.path.join(args.dataset_path, 'sparse-%s-%s' % (args.method_name, 'ref' if refine else 'raw'))

    if os.path.exists(paths.database_path):
        raise FileExistsError('Database file already exists.')
    shutil.copy(
        os.path.join(args.dataset_path, 'database.db'),
        paths.database_path
    )

    matching_stats = import_matches(
        args.colmap_path, args.method_name,
        paths.database_path, paths.image_path, paths.match_list_path,
        args.matches_file, args.solution_file
    )
    reconstruction_stats = reconstruct(
        args.colmap_path,
        paths.database_path, paths.image_path, paths.sparse_path
    )

    with open(args.output_file, 'w') as f:
        f.write(json.dumps(matching_stats))
        f.write('\n')
        f.write(json.dumps(reconstruction_stats))


if __name__ == '__main__':
    main()
