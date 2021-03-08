# Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu (mihai.dusmanu@inf.ethz.ch)

import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_positions(shape, device):
    h, w = shape

    rows = torch.linspace(-1, 1, h, device=device).view(h, 1).repeat(1, w)
    columns = torch.linspace(-1, 1, w, device=device).view(1, w).repeat(h, 1)

    grid = torch.stack([rows, columns], dim=-1)

    return grid


def extract_patches(image, ij, device, patch_size=33):
    image = torch.tensor(image).float().to(device).permute([2, 0, 1])
    c, h, w = image.size()

    grid_patch = grid_positions([patch_size, patch_size], device)  # ps x ps x 2
    grid_patch[:, :, 0] *= patch_size / (h - 1)
    grid_patch[:, :, 1] *= patch_size / (w - 1)

    norm_ij = torch.tensor(ij).float().to(device)
    norm_ij[:, 0] = norm_ij[:, 0] / (h - 1) * 2 - 1
    norm_ij[:, 1] = norm_ij[:, 1] / (w - 1) * 2 - 1

    full_ij = norm_ij.view(-1, 1, 1, 2) + grid_patch

    patches = F.grid_sample(
        image.unsqueeze(0), full_ij[:, :, :, [1, 0]].reshape(1, -1, patch_size, 2),
        padding_mode='reflection', align_corners=True
    ).squeeze(0)
    patches = patches.view(c, -1, patch_size, patch_size).permute([1, 0, 2, 3])

    return patches.cpu()


def estimate_displacements(reference_batch, target_batch, net, device, batch_size=1024, symmetric=False):
    reference_batch = net.normalize_batch(reference_batch)
    target_batch = net.normalize_batch(target_batch)

    n_batches = reference_batch.shape[0] // batch_size + (reference_batch.shape[0] % batch_size != 0)

    if symmetric:
        displacements12 = torch.zeros([0, 2]).to(device)
        displacements21 = torch.zeros([0, 2]).to(device)
        for batch_idx in range(n_batches):
            c_displacements12, c_displacements21 = net.forward_sym(
                reference_batch[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device),
                target_batch[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)
            )
            displacements12 = torch.cat([displacements12, c_displacements12], dim=0)
            displacements21 = torch.cat([displacements21, c_displacements21], dim=0)

        return displacements12, displacements21
    else:
        displacements = torch.zeros([0, 2]).to(device)
        for batch_idx in range(n_batches):
            c_displacements = net.forward(
                reference_batch[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device),
                target_batch[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)
            )
            displacements = torch.cat([displacements, c_displacements], dim=0)

        return displacements


def extract_patches_and_estimate_displacements(
        image1, ij1,
        image2, ij2,
        net, device, batch_size,
        symmetric=True, grid=True, octave=0
):
    if grid:
        grid_ij = np.stack(np.meshgrid([-1., 0, 1.], [-1., 0, 1.], indexing='ij'), axis=-1).reshape(-1, 2) * 8
        # Scale to current octave.
        grid_ij /= (2 ** octave)
    else:
        grid_ij = np.array([[0., 0.]])
    all_ij1 = (ij1[:, np.newaxis] + grid_ij[np.newaxis]).reshape(-1, 2)
    all_ij2 = (ij2[:, np.newaxis] + grid_ij[np.newaxis]).reshape(-1, 2)

    batch1 = extract_patches(
        image1, all_ij1, device, patch_size=33
    )

    batch2 = extract_patches(
        image2, all_ij2, device, patch_size=33
    )

    if symmetric:
        displacements12, displacements21 = estimate_displacements(
            batch1, batch2, net, device, batch_size=batch_size, symmetric=symmetric
        )

        if grid:
            grid_displacements12 = displacements12.reshape(-1, 3, 3, 2)
            grid_displacements21 = displacements21.reshape(-1, 3, 3, 2)
            return grid_displacements12.cpu().numpy(), grid_displacements21.cpu().numpy()
        else:
            return displacements12.cpu().numpy(), displacements21.cpu().numpy()
    else:
        displacements12 = estimate_displacements(
            batch1, batch2, net, device, batch_size=batch_size, symmetric=symmetric
        )

        if grid:
            grid_displacements12 = displacements12.reshape(-1, 3, 3, 2)
            return grid_displacements12.cpu().numpy()
        else:
            return displacements12.cpu().numpy()


def refine_matches_coarse_to_fine(
        image1, keypoints1,
        image2, keypoints2,
        matches,
        net, device, batch_size,
        symmetric=True, grid=True
):
    ij1 = keypoints1[matches[:, 0]][:, [1, 0]]
    ij2 = keypoints2[matches[:, 1]][:, [1, 0]]

    if symmetric:
        # Coarse refinement.
        coarse_displacements12, coarse_displacements21 = extract_patches_and_estimate_displacements(
            image1, ij1,
            image2, ij2,
            net, device, batch_size,
            symmetric=symmetric, grid=False, octave=0.
        )

        # Fine refinement.
        up_image1 = cv2.pyrUp(image1)
        up_image2 = cv2.pyrUp(image2)

        displacements12 = .5 * extract_patches_and_estimate_displacements(
            up_image1, 2. * ij1,
            up_image2, 2. * (ij2 + coarse_displacements12 * 16),
            net, device, batch_size,
            symmetric=False, grid=grid, octave=-1.
        )
        displacements21 = .5 * extract_patches_and_estimate_displacements(
            up_image2, 2. * ij2,
            up_image1, 2. * (ij1 + coarse_displacements21 * 16),
            net, device, batch_size,
            symmetric=False, grid=grid, octave=-1.
        )

        if grid:
            return coarse_displacements12[:, np.newaxis, np.newaxis] + displacements12, coarse_displacements21[:, np.newaxis, np.newaxis] + displacements21
        else:
            return coarse_displacements12 + displacements12, coarse_displacements21 + displacements21
    else:
        # Coarse refinement.
        coarse_displacements12 = extract_patches_and_estimate_displacements(
            image1, ij1,
            image2, ij2,
            net, device, batch_size,
            symmetric=symmetric, grid=False, octave=0.
        )

        # Fine refinement.
        up_image1 = cv2.pyrUp(image1)
        up_image2 = cv2.pyrUp(image2)

        displacements12 = .5 * extract_patches_and_estimate_displacements(
            up_image1, 2. * ij1,
            up_image2, 2. * (ij2 + coarse_displacements12 * 16),
            net, device, batch_size,
            symmetric=False, grid=grid, octave=-1.
        )

        if grid:
            return coarse_displacements12[:, np.newaxis, np.newaxis] + displacements12
        else:
            return coarse_displacements12 + displacements12
