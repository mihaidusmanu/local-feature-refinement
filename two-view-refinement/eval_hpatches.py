import argparse

import cv2

import numpy as np

import os

import sys

import torch

from tqdm import tqdm

import types

from feature_matchers import mnn_similarity_matcher, mnn_ratio_matcher

from model import PANet

from refinement import refine_matches_coarse_to_fine as refine_matches

max_size_dict = {
    'sift': (1600, 3200),
    'surf': (1600, 3200),
    'd2-net': (1600, 2800),
    'keynet': (1600, 3200),
    'r2d2': (1600, 3200),
    'superpoint': (1600, 2800),
}


# (
# type of matcher,
# matcher threshold
# )
matcher_dict = {
    'sift': ('ratio', 0.8),
    'surf': ('ratio', 0.8),
    'd2-net': ('similarity', 0.8),
    'keynet': ('ratio', 0.9),
    'r2d2': ('similarity', 0.9),
    'superpoint': ('similarity', 0.755)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--image_path', type=str, required=True,
    #     help='path to the images'
    # )
    parser.add_argument(
        '--dataset_name', type=str, required=True,
        help='name of the dataset'
    )
    # parser.add_argument(
    #     '--max_edge', type=int, defaul=True,
    #     help='maximum image size at feature extraction octave 0'
    # )
    # parser.add_argument(
    #     '--max_sum_edges', type=int, required=True,
    #     help='maximum sum of image sizes at feature extraction octave 0'
    # )

    # parser.add_argument(
    #     '--match_list_file', type=str, required=True,
    #     help='matching list file'
    # )

    parser.add_argument(
        '--method_name', type=str, required=True,
        help='name of the method'
    )

    # parser.add_argument(
    #     '--output_file', type=str, required=True,
    #     help='output file'
    # )

    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size'
    )

    # parser.add_argument(
    #     '--matcher', type=str, required=True,
    #     help='matcher (possible values: "similarity", "ratio")'
    # )
    # parser.add_argument(
    #     '--threshold', type=float, required=True,
    #     help='threshold for matchers (either Lowe\'s ratio threshold or similarity threshold)'
    # )
    args = parser.parse_args()

    # Define extra paths.
    paths = types.SimpleNamespace()
    paths.dataset_path = os.path.join('LFE', args.dataset_name)
    paths.image_path = os.path.join(paths.dataset_path, 'images')
    paths.match_list_file = os.path.join(paths.dataset_path, 'match-list.txt')
    paths.matches_file = os.path.join('output', '%s-%s-matches.pb' % (args.method_name, args.dataset_name))
    paths.solution_file = os.path.join('output', '%s-%s-solution.pb' % (args.method_name, args.dataset_name))
    paths.ref_results_file = os.path.join('output', '%s-%s-ref.txt' % (args.method_name, args.dataset_name))
    paths.raw_results_file = os.path.join('output', '%s-%s-raw.txt' % (args.method_name, args.dataset_name))


    def read_feats(seq_name, img_idx):
        # Load the features.
        features = np.load(os.path.join(paths.dataset_path,
            seq_name, '%s.%s.%s' % (img_idx, 'ppm', args.method_name)
        ), allow_pickle=True)
        keypoints = features['keypoints'][:, : 2]
        descriptors = features['descriptors']

        # Downscale keypoints to the feature extraction resolution.
        max_edge = max_size_dict[args.method_name][0]
        max_sum_edges = max_size_dict[args.method_name][1]
        image = cv2.imread(os.path.join(paths.dataset_path, seq_name, str(img_idx) + '.ppm'))
        fact = max(1, max(image.shape) / max_edge, sum(image.shape[: -1]) / max_sum_edges)
        keypoints *= 1 / fact

        return keypoints, descriptors

    
    # Torch settings for the matcher & two-view estimation network.
    torch.set_grad_enabled(False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # hyper parameters for hpatches
    n_i = 52
    n_v = 56
    matcher = str(matcher_dict[args.method_name][0])
    threshold = matcher_dict[args.method_name][1]

    n_feats = []
    n_matches = []
    seq_type = []
    lim = [1, 15]
    rng = np.arange(lim[0], lim[1] + 1)
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    seq_names = sorted(os.listdir(paths.dataset_path))
    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            if matcher == 'similarity':
                matches, sim = mnn_similarity_matcher(
                    torch.from_numpy(descriptors_a).float().to(device=device), 
                    torch.from_numpy(descriptors_b).float().to(device=device),
                    ratio=threshold
                )
            elif matcher == 'ratio':
                matches, sim = mnn_ratio_matcher(
                    torch.from_numpy(descriptors_a).float().to(device=device), 
                    torch.from_numpy(descriptors_b).float().to(device=device),
                    ratio=threshold
                )
            else:
                raise NotImplementedError
            
            homography = np.loadtxt(os.path.join(paths.dataset_path, seq_name, "H_1_" + str(im_idx)))
            
            # print('matches shape: ',matches.shape)
            # print('keypoint shape: ', keypoints_a.shape)
            pos_a = keypoints_a[matches[:, 0], : 2] 
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])
            
            if dist.shape[0] == 0:
                dist = np.array([float("inf")])
            
            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
    
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    # error summary
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5), 
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )