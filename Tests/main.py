import argparse
from evaluation import run_evaluation
from absolute_error_evaluation import evaluate_camera_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type = str,
        required = False,
        default="../datasets/ETH3D/courtyard/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../results/glomap/courtyard/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    args = parser.parse_args()

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    run_evaluation(gt_model_path=gt_model_path, est_model_path=est_model_path)

