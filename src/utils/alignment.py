import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation

def kabsch(pts1, pts2, estimate_scale=False) -> Tuple[np.ndarray, float]:
    """
    Run the Kabsch algorithm to estimate the alignment between the estimated and ground truth poses.
    """
    c_pts1 = pts1 - pts1.mean(axis=0)
    c_pts2 = pts2 - pts2.mean(axis=0)

    covariance = np.matmul(c_pts1.T, c_pts2) / c_pts1.shape[0]

    U, S, VT = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(np.matmul(VT.T, U.T)))
    correction = np.eye(3)
    correction[2, 2] = d

    if estimate_scale:
        pts_var = np.mean(np.linalg.norm(c_pts2, axis=1) ** 2)
        scale_factor = pts_var / np.trace(S * correction)
    else:
        scale_factor = 1.

    R = scale_factor * np.matmul(np.matmul(VT.T, correction), U.T)
    t = pts2.mean(axis=0) - np.matmul(R, pts1.mean(axis=0))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, scale_factor

def get_inliers(T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):
    """
    Calculate the inliers based on the estimated transformation.
    """
    # T aligns ground truth poses with estimates poses
    poses_gt_transformed = np.array([T @ pose for pose in poses_gt])

    # Calculate differences in position and rotations
    translations_delta = np.array([pose_gt[:3, 3] - pose_est[:3, 3] for pose_gt, pose_est in zip(poses_gt_transformed, poses_est)])
    rotations_delta = np.array([pose_gt[:3, :3] @ pose_est[:3, :3].T for pose_gt, pose_est in zip(poses_gt_transformed, poses_est)])

    # translation inliers
    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    # rotation inliers
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (inlier_threshold_r / 180 * np.pi)
    # intersection of both
    return np.logical_and(inliers_r, inliers_t)

def ransac_kabsch(
        est_poses: list,
        gt_poses: list,
        estimate_scale: bool = True,
        inlier_threshold_r: float = 5,
        inlier_threshold_t: float = 0.1,
        confidence: float = 0.99,
        max_iterations: int = 10000
) -> Tuple[np.ndarray, float]:
    """
    Estimate the alignment between the estimated and ground truth poses using
    LO-RANSAC and Kabsch algorithm as model estimation.
    """
    best_model = None
    best_score = 0
    best_scale = 1.0
    sample_size = 3

    for _ in range(max_iterations):
        # Randomly sample 3 pairs of poses
        sample_indices = np.random.choice(len(est_poses), sample_size, replace=False)
        est_sample = [est_poses[idx] for idx in sample_indices]
        gt_sample = [gt_poses[idx] for idx in sample_indices]

        # Estimate the alignment using Kabsch algorithm
        T, scale = kabsch(
            np.array([pose[:3, 3] for pose in est_sample]),
            np.array([pose[:3, 3] for pose in gt_sample]),
            estimate_scale=estimate_scale
        )

        # Calculate the inliers based on the estimated transformation
        inliers = get_inliers(T, gt_poses, est_poses, inlier_threshold_t, inlier_threshold_r)
        score = len(inliers) / len(est_poses)

        if score > best_score:
            candidate_T = T
            candidate_scale = scale
            candidate_inliers = inliers
            candidate_score = score

            # Local optimization
            for _ in range(10):
                inlier_indices = np.where(candidate_inliers)[0]
                if len(inlier_indices) < sample_size:
                    break
                # Re-estimate transformation using all inlier correspondences
                est_inliers = [est_poses[i] for i in inlier_indices]
                gt_inliers = [gt_poses[i] for i in inlier_indices]
                T_refined, scale_refined = kabsch(
                    np.array([pose[:3, 3] for pose in est_inliers]),
                    np.array([pose[:3, 3] for pose in gt_inliers]),
                    estimate_scale=estimate_scale
                )
                new_inliers = get_inliers(T_refined, gt_poses, est_poses, inlier_threshold_t, inlier_threshold_r)
                new_score = np.sum(new_inliers) / len(est_poses)

                if new_score > candidate_score:
                    candidate_T, candidate_scale = T_refined, scale_refined
                    candidate_inliers, candidate_score = new_inliers, new_score

            # Update best model if local optimization improved the score
            if candidate_score > best_score:
                best_model, best_scale, best_score = candidate_T, candidate_scale, candidate_score

                # Update max_iterations based on the new inlier ratio
                w = candidate_score  # fraction of inliers
                eps = np.finfo(float).eps  # prevent division by zero
                max_iterations = min(
                    max_iterations,
                    np.log(1 - confidence) / (np.log(1 - max(w, eps) ** sample_size) + eps)
                )

    return best_model, best_scale


