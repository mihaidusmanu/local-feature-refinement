# Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu (mihai.dusmanu@inf.ethz.ch)

import argparse

import cv2

import numpy as np

import os

import sys

import torch

from tqdm import tqdm


from feature_matchers import mnn_similarity_matcher, mnn_ratio_matcher

from model import PANet

from refinement import refine_matches_coarse_to_fine as refine_matches

from types_pb2 import MatchingFile


# Debug flag.
skip_refinement = ('SKIP_REFINEMENT' in os.environ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_path', type=str, required=True,
        help='path to the images'
    )
    parser.add_argument(
        '--max_edge', type=int, required=True,
        help='maximum image size at feature extraction octave 0'
    )
    parser.add_argument(
        '--max_sum_edges', type=int, required=True,
        help='maximum sum of image sizes at feature extraction octave 0'
    )

    parser.add_argument(
        '--match_list_file', type=str, required=True,
        help='matching list file'
    )

    parser.add_argument(
        '--method_name', type=str, required=True,
        help='name of the method'
    )

    parser.add_argument(
        '--output_file', type=str, required=True,
        help='output file'
    )

    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size'
    )

    parser.add_argument(
        '--matcher', type=str, required=True,
        help='matcher (possible values: "similarity", "ratio")'
    )
    parser.add_argument(
        '--threshold', type=float, required=True,
        help='threshold for matchers (either Lowe\'s ratio threshold or similarity threshold)'
    )
    args = parser.parse_args()

    # Protobuf memory limitation avoidance.
    dump_interval = 5000

    # Torch settings for the matcher & two-view estimation network.
    torch.set_grad_enabled(False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create the two-view estimation network.
    net = PANet().to(device)

    # Read the matching list.
    with open(args.match_list_file, 'r') as f:
        raw_match_list = f.readlines()

    # Match and estimate displacements.
    matching_file_proto = MatchingFile()
    part_idx = -1
    image_name1 = None
    for pair_idx, raw_match in enumerate(tqdm(raw_match_list, total=len(raw_match_list), file=sys.__stdout__)):
        image_name1_, image_name2 = raw_match.strip('\n').split(' ')

        if image_name1_ != image_name1:
            image_name1 = image_name1_
            image1 = cv2.imread(os.path.join(args.image_path, image_name1))
            image1 = image1[:, :, ::-1]  # BGR -> RGB
            if len(image1.shape) == 2:
                image1 = image1[:, :, np.newaxis].repeat(3, axis=-1)
            fact1 = max(1, max(image1.shape) / args.max_edge, sum(image1.shape[: -1]) / args.max_sum_edges)
            image1 = cv2.resize(image1, None, fx=(1 / fact1), fy=(1 / fact1), interpolation=cv2.INTER_AREA)

        image2 = cv2.imread(os.path.join(args.image_path, image_name2))
        image2 = image2[:, :, ::-1]  # BGR -> RGB
        if len(image2.shape) == 2:
            image2 = image2[:, :, np.newaxis].repeat(3, axis=-1)
        fact2 = max(1, max(image2.shape) / args.max_edge, sum(image2.shape[: -1]) / args.max_sum_edges)
        image2 = cv2.resize(image2, None, fx=(1 / fact2), fy=(1 / fact2), interpolation=cv2.INTER_AREA)

        # Load the features.
        features1 = np.load(os.path.join(
            args.image_path, '%s.%s' % (image_name1, args.method_name)
        ), allow_pickle=True)
        keypoints1 = features1['keypoints'][:, : 2]
        descriptors1 = features1['descriptors']

        features2 = np.load(os.path.join(
            args.image_path, '%s.%s' % (image_name2, args.method_name)
        ), allow_pickle=True)
        keypoints2 = features2['keypoints'][:, : 2]
        descriptors2 = features2['descriptors']

        if keypoints1.shape[0] > 0 and keypoints2.shape[0] > 0:
            # Downscale keypoints to the feature extraction resolution.
            keypoints1 *= 1 / fact1
            keypoints2 *= 1 / fact2

            # Feature matching.
            if args.matcher == 'similarity':
                matches, sim = mnn_similarity_matcher(
                    torch.tensor(descriptors1).float().to(device),
                    torch.tensor(descriptors2).float().to(device),
                    threshold=args.threshold
                )
            elif args.matcher == 'ratio':
                matches, sim = mnn_ratio_matcher(
                    torch.tensor(descriptors1).float().to(device),
                    torch.tensor(descriptors2).float().to(device),
                    ratio=args.threshold
                )
            else:
                raise NotImplementedError

            # Keypoint refinement.
            if skip_refinement:
                grid_displacements12 = np.zeros((matches.shape[0], 3, 3, 2))
                grid_displacements21 = np.zeros((matches.shape[0], 3, 3, 2))
            else:
                grid_displacements12, grid_displacements21 = refine_matches(
                    image1, keypoints1,
                    image2, keypoints2,
                    matches,
                    net, device, args.batch_size, symmetric=True, grid=True
                )
        else:
            matches = np.zeros((0, 2))

        # Build the proto object.
        image_pair_proto = matching_file_proto.image_pairs.add()

        image_pair_proto.image_name1 = image_name1
        image_pair_proto.fact1 = fact1
        image_pair_proto.image_name2 = image_name2
        image_pair_proto.fact2 = fact2

        for match_idx in range(matches.shape[0]):
            match_proto = image_pair_proto.matches.add()

            match_proto.feature_idx1 = matches[match_idx, 0]
            match_proto.feature_idx2 = matches[match_idx, 1]

            match_proto.similarity = sim[match_idx]

            for grid_i in range(3):
                for grid_j in range(3):
                    disp_proto = match_proto.disp1.add()
                    disp_proto.di = grid_displacements21[match_idx, grid_i, grid_j, 0]
                    disp_proto.dj = grid_displacements21[match_idx, grid_i, grid_j, 1]

                    disp_proto = match_proto.disp2.add()
                    disp_proto.di = grid_displacements12[match_idx, grid_i, grid_j, 0]
                    disp_proto.dj = grid_displacements12[match_idx, grid_i, grid_j, 1]

        if pair_idx % dump_interval == dump_interval - 1:
            part_idx += 1
            output_file = open('%s.part.%d' % (args.output_file, part_idx), 'wb')
            output_file.write(matching_file_proto.SerializeToString())
            output_file.close()
            matching_file_proto = MatchingFile()

    # Save the proto object to disk.
    if part_idx == -1:
        output_file = open(args.output_file, 'wb')
        output_file.write(matching_file_proto.SerializeToString())
        output_file.close()
    else:
        part_idx += 1
        output_file = open('%s.part.%d' % (args.output_file, part_idx), 'wb')
        output_file.write(matching_file_proto.SerializeToString())
        output_file.close()
