import argparse

import cv2

import numpy as np

import os

import shutil

import torch

import types

from tqdm import tqdm

from colmap_utils import compute_reconstruction_statistics, localize

from model import PANet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', type=str, required=True,
        help='name of the dataset'
    )
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
        '--refine', dest='refine', action='store_true',
        help='use refinement'
    )
    parser.set_defaults(refine=False)

    parser.add_argument(
        '--max_edge', type=int, required=True,
        help='maximum image size at feature extraction octave 0'
    )
    parser.add_argument(
        '--max_sum_edges', type=int, required=True,
        help='maximum sum of image sizes at feature extraction octave 0'
    )

    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size'
    )

    parser.add_argument(
        '--output_path', type=str, default=None,
        help='path to the output results file'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
   # Torch settings for the two-view estimation network.
    torch.set_grad_enabled(False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create the two-view estimation network.
    net = PANet().to(device)

    # Parse arguments.
    args = parse_args()

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.dummy_database_path = os.path.join(
        args.dataset_path, 'database.db'
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

    # Recover covisibility from reference model.
    images, intrinsics, poses, n_covisible_points = compute_reconstruction_statistics(paths.reference_model_path)

    # Invert the image name to image id dictionary.
    image_id_to_image_name = {}
    for image_name, image_id in images.items():
        image_id_to_image_name[image_id] = image_name

    # Precompute downsized images and scaling factors.
    numpy_images = {}
    facts = {}
    for image_name in images:
        image = cv2.imread(os.path.join(paths.image_path, image_name))
        image = image[:, :, ::-1]  # BGR -> RGB
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=-1)
        fact = max(1, max(image.shape) / args.max_edge, sum(image.shape[: -1]) / args.max_sum_edges)
        image = cv2.resize(image, None, fx=(1 / fact), fy=(1 / fact), interpolation=cv2.INTER_AREA)
        numpy_images[image_name] = image
        facts[image_name] = fact

    # Randomly select 10 query images.
    np.random.seed(0)
    all_images = list(images.items())
    selected_images = [all_images[idx] for idx in np.random.choice(len(images), 10, replace=False)]

    # Evaluation loop.
    for image_name, image_id in tqdm(selected_images):
        # Recover annotations.
        annotated_pose = poses[image_name]
        camera_parameters = intrinsics[image_name]
        camera_dict = {
            'model': camera_parameters[0],
            'width': int(camera_parameters[1]),
            'height': int(camera_parameters[2]),
            'params': list(map(float, camera_parameters[3 :]))
        }

        # Find 3 nearby images.
        holdout_image_ids = np.argsort(n_covisible_points[image_id, :])[:: -1][: 3]
        holdout_image_names = []
        for image_id_ in holdout_image_ids:
            if n_covisible_points[image_id, image_id_] == 0:
                continue
            assert(image_id_ in image_id_to_image_name)
            holdout_image_names.append(image_id_to_image_name[image_id_])

        # Localize image.
        pose = localize(
            image_id, image_name, camera_dict, holdout_image_names, numpy_images, facts, net, device, args.batch_size,
            args.colmap_path, args.dataset_name, args.dataset_path, args.method_name, args.refine, args.matches_file,
            paths.dummy_database_path, paths.image_path, paths.reference_model_path, paths.match_list_path
        )

        # Pose error.
        with open(args.output_path, 'a') as f:
            if pose is None:
                f.write('[%s] - failed\n' % image_name)
            else:
                # Compute the error
                annotated_R = annotated_pose[: 3, : 3]
                annotated_t = annotated_pose[: 3, 3]
                R = pose[: 3, : 3]
                t = pose[: 3, 3]

                rotation_difference = R @ annotated_R.transpose()
                ori_error = np.rad2deg(np.arccos(np.clip((np.trace(rotation_difference) - 1) / 2, -1, 1)))

                annotated_C = (-1) * annotated_R.transpose() @ annotated_t
                C = (-1) * R.transpose() @ t
                center_error = np.linalg.norm(C - annotated_C)

                f.write('[%s] - orientation error %4f deg, camera center error %4f\n' % (image_name, ori_error, center_error))