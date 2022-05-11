import argparse
import os.path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import sys

# sys.path.insert(0, ".")
from ibug.face_pose_augmentation import FacePoseAugmentor, TDDFAPredictor


def read_landmark_file(file_path):
    with open(file_path, mode='r') as f:
        lines = f.readlines()
    num_landmarks = int(lines[0])

    landmarks = np.zeros(shape=(num_landmarks, 2))

    for i in range(1, len(lines)):
        lm_str = lines[i].split(' ')
        landmarks[i - 1] = np.array([float(lm_str[0]), float(lm_str[1])])

    return landmarks


def augment_thread(file_name: str, data_root: str, output_folder: str, tddfa_predictor: TDDFAPredictor,
                   face_pose_augmentor: FacePoseAugmentor,
                   delta_poses: np.ndarray):
    """

    Args:
        file_name: original image name
        data_root: Path to original data folder
        output_folder: Path to output folder
        tddfa_predictor: TDDFA Predictor
        face_pose_augmentor: Face Pose Augmentor
        delta_poses: should be a Nx3 array, each row giving the delta pitch, delta yaw, and delta roll of a target pose.

    Returns:

    """
    images_folder = os.path.join(data_root, 'images')
    labels_folder = os.path.join(data_root, 'labels')
    landmarks_folder = os.path.join(data_root, 'landmarks')

    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')

    # Read original image, landmark and label
    original_image = cv2.imread(os.path.join(images_folder, file_name))
    original_landmark = read_landmark_file(os.path.join(landmarks_folder, file_name.replace(".jpg", ".txt")))
    original_label = cv2.imread(os.path.join(labels_folder, file_name.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED)

    # Run TDDFA
    tddfa_result = TDDFAPredictor.decode(tddfa_predictor(original_image, original_landmark, rgb=False))[0]

    # Run augmentation
    image_augmentation_result = face_pose_augmentor(original_image, tddfa_result, delta_poses, None)
    label_augmentation_result = face_pose_augmentor(original_label, tddfa_result, delta_poses, None)

    # Save output
    for i in range(delta_poses.shape[0]):
        image_augmented = image_augmentation_result[i]['warped_image']
        label_augmented = label_augmentation_result[i]['warped_image']

        output_image_name = file_name.replace(".jpg", f"_{i}.jpg")
        cv2.imwrite(os.path.join(output_images_folder, output_image_name), image_augmented)

        output_label_name = file_name.replace(".jpg", f"_{i}.png")
        cv2.imwrite(os.path.join(output_labels_folder, output_label_name), label_augmented)


def main(args):
    # Input and output folders
    data_root = args.data_root
    images_folder = os.path.join(data_root, 'images')

    output_folder = args.output_folder
    assert not os.path.exists(output_folder), "Output folder existed!"
    os.makedirs(output_folder)
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    os.makedirs(output_images_folder)
    os.makedirs(output_labels_folder)

    # Init executor
    num_workers = args.num_workers

    executor = ThreadPoolExecutor(max_workers=num_workers)

    # Init tddfa predictor
    tddfa = TDDFAPredictor(device="cuda:0")

    # Init augmentor
    augmentor = FacePoseAugmentor()

    # Delta poses
    delta_poses = np.array([
        [30 / 180, 0, 0],
        [60 / 180, 0, 0],
        [90 / 180, 0, 0],
        [120 / 180, 0, 0],
        [150 / 180, 0, 0],
    ])

    # Run
    for r, d, f in os.walk(images_folder):
        for file in f:
            if ".jpg" in file:
                executor.submit(augment_thread, file, data_root, output_folder, tddfa, augmentor, delta_poses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LaPa augmentation")
    parser.add_argument("--data_root",
                        type=str,
                        help="Path to data folder")
    parser.add_argument("--output_folder",
                        type=str,
                        help="Path to output folder")
    parser.add_argument("--num_workers",
                        type=int,
                        default=6,
                        help="Number of workers")
    args = parser.parse_args()
    main(args)
