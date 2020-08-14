import argparse

import os

import shutil

import types

from colmap_utils import generate_empty_reconstruction, import_features, triangulate


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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    refine = (args.solution_file is not None)

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(
        args.dataset_path, '%s-%s.db' % (args.method_name, 'ref' if refine else 'raw')
    )
    paths.image_path = os.path.join(
        args.dataset_path, 'images'
    )
    paths.reference_model_path = os.path.join(
        args.dataset_path, 'dslr_calibration_undistorted'
    )
    paths.match_list_path = os.path.join(
        args.dataset_path, 'match-list.txt'
    )
    paths.empty_model_path = os.path.join(
        args.dataset_path, 'sparse-%s-%s-empty' % (args.method_name, 'ref' if refine else 'raw')
    )
    paths.model_path = os.path.join(
        args.dataset_path, 'sparse-%s-%s' % (args.method_name, 'ref' if refine else 'raw')
    )
    paths.ply_model_path = os.path.join(
        args.dataset_path, 'sparse-%s-%s.ply' % (args.method_name, 'ref' if refine else 'raw')
    )

    # Create a copy of the dummy database.
    if os.path.exists(paths.database_path):
        raise FileExistsError(
            'The database file already exists.'
        )
    shutil.copyfile(
        os.path.join(args.dataset_path, 'database.db'),
        paths.database_path
    )

    # Reconstruction pipeline.
    _ = generate_empty_reconstruction(
        paths.reference_model_path,
        paths.empty_model_path
    )
    import_features(
        args.colmap_path, args.method_name,
        paths.database_path, paths.image_path, paths.match_list_path,
        args.matches_file, args.solution_file
    )
    triangulate(
        args.colmap_path,
        paths.database_path, paths.image_path,
        paths.empty_model_path,
        paths.model_path, paths.ply_model_path
    )
