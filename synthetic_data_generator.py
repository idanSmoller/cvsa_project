import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
from numpy import pi
import argparse
import random
import os
import json
import shutil
import glob
import cv2
from PIL import Image
import debugpy


NUM_IMAGES = 20

BASE_PATH = "/datashare/project"
TOOLS_DEFAULT_FOLDER = os.path.join(BASE_PATH, "surgical_tools_models")
CAMERA_PARAMS_DEFAULT = os.path.join(BASE_PATH, "camera.json")

NUM_OF_NEEDLE_HOLDERS_VERSIONS = len(glob.glob(os.path.join(TOOLS_DEFAULT_FOLDER, "needle_holder", "*.obj")))
NUM_OF_TWEEZERS_VERSIONS = len(glob.glob(os.path.join(TOOLS_DEFAULT_FOLDER, "tweezers", "*.obj")))
print(f"Number of needle holders versions: {NUM_OF_NEEDLE_HOLDERS_VERSIONS}")
print(f"Number of tweezers versions: {NUM_OF_TWEEZERS_VERSIONS}")
RIGHT_HAND_FILE = "right_hand.obj"
OUTPUT_DIR_DEFAULT = "synthetic_data"
BACKGROUND_PATH = "background.png"


def load_objects(tools_base_folder, right_hand_path):
    """
    Load surgical tool objects from the specified folder.
    Args:
        tools_base_folder (str): Path to the folder containing surgical tool models.
        right_hand_path (str): Path to the right hand object file.
    Returns:
        tuple: A tuple containing lists of needle holders and tweezers, the right and left hand objects, and a light object.
    """
    needle_holders = []
    for i in range(1, NUM_OF_NEEDLE_HOLDERS_VERSIONS + 1):
        needle_holder = bproc.loader.load_obj(os.path.join(tools_base_folder, "needle_holder", f"NH{i}.obj"))[0]
        needle_holder.set_cp("category_id", 1)
        needle_holder.set_cp("cp_physics", False)
        needle_holder.set_scale([1.5, 1.5, 1.5])
        needle_holder.set_name(f"NeedleHolder")
        needle_holder.hide()
        needle_holders.append(needle_holder)

    tweezers = []
    for i in range(1, NUM_OF_TWEEZERS_VERSIONS + 1):
        tweezer = bproc.loader.load_obj(os.path.join(tools_base_folder, "tweezers", f"T{i}.obj"))[0]
        tweezer.set_cp("category_id", 2)
        tweezer.set_cp("cp_physics", False)
        tweezer.set_scale([1.5, 1.5, 1.5])
        tweezer.set_name(f"Tweezers")
        tweezer.hide()
        tweezers.append(tweezer)

    right_hand = bproc.loader.load_obj(right_hand_path)[0]
    right_hand.set_cp("cp_physics", False)
    right_hand.set_cp("category_id", 0)
    right_hand.set_scale([14, 14, 14])
    right_hand.set_name("RightHand")

    left_hand = right_hand.duplicate()
    left_hand.set_cp("cp_physics", False)
    left_hand.set_cp("category_id", 0)
    left_hand.set_scale([-14, 14, 14])
    left_hand.set_name("LeftHand")

    light = bproc.types.Light()
    light.set_type("POINT")

    return needle_holders, tweezers, right_hand, left_hand, light


def load_camera_params(camera_params_path):
    """
    Load camera parameters from a JSON file.
    Args:
        camera_params_path (str): Path to the camera parameters JSON file.
    Returns:
        dict: Camera parameters loaded from the file.
    """
    # Set camera intrinsics parameters
    with open(camera_params_path, "r") as file:
        camera_params = json.load(file)

    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    im_width = camera_params["width"]
    im_height = camera_params["height"]
    K = np.array([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]])
    CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height) 
    camera_location = [0, 0, 20]
    rotation_matrix = bproc.camera.rotation_from_forward_vec([-x for x in camera_location], inplane_rot=0)
    cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

    return cam2world_matrix, im_width, im_height


def set_frame_positions(needle_holders, tweezers, right_hand, left_hand, light):
    """
    Choose a random version of the needle holder and tweezer, and set their positions and rotations randomly.
    Args:
        needle_holders (list): List of needle holder objects.
        tweezers (list): List of tweezer objects.
        right_hand (bproc.types.Object): Right hand object.
        left_hand (bproc.types.Object): Left hand object.
        light (bproc.types.Light): Light object to set its position and energy.
    Returns:
        tuple: The chosen needle holder and tweezer objects (returned to hide after rendering).
    """
    light.set_location(np.random.uniform([8, -2, 0], [12, 6, 5]))
    light.set_energy(np.random.uniform(2000, 5000))

    needle_holder_version = random.randint(1, NUM_OF_NEEDLE_HOLDERS_VERSIONS)
    needle_holder = needle_holders[needle_holder_version - 1]
    needle_holder.hide(False)
    needle_holder_location = np.random.uniform([-3, -1.2, -1], [-1, 0.2, 1])
    needle_holder_rotation = np.random.uniform([0, 0, (-5/8)*pi], [0, (1/2)*pi, (-1/4)*pi])
    needle_holder.set_rotation_euler(needle_holder_rotation)
    needle_holder.set_location(needle_holder_location)

    right_hand_location = needle_holder_location + \
        np.array([-0.6 + 0.02 * needle_holder_version, 0.5, 0.5 + 0.1 * needle_holder_version]) + \
        np.random.uniform([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])
    right_hand_rotation = needle_holder_rotation + \
        np.array([0, 0, -(1/2)*pi]) + \
        np.random.uniform([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])
    right_hand.set_rotation_euler(right_hand_rotation)
    right_hand.set_location(right_hand_location)

    tweezer_version = random.randint(1, NUM_OF_TWEEZERS_VERSIONS)
    tweezer = tweezers[tweezer_version - 1]
    tweezer.hide(False)
    tweezer_location = np.random.uniform([-1, -0.7, -1], [-0.5, 1.5, 1])
    tweezer_rotation = np.random.uniform([0, 0, (10/12)*pi], [0, (1/2)*pi, (13/12)*pi])
    tweezer.set_rotation_euler(tweezer_rotation)
    tweezer.set_location(tweezer_location)

    left_hand_location = tweezer_location + \
        np.array([-0.5, 0.6, 0.5]) + \
        np.random.uniform([0, -0.1, -0.1], [0.1, 0.1, 0.1])
    left_hand_rotation = tweezer_rotation + \
        np.array([(1/2)*pi, -(1/2)*pi, 0]) + \
        np.random.uniform([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])
    left_hand.set_rotation_euler(left_hand_rotation)
    left_hand.set_location(left_hand_location)

    return needle_holder, tweezer


def get_random_background(background_path, im_width, im_height):
    """
    Get a random background image.
    Args:
        background_path (str): Path to the background image.
        im_width (int): Width of the image.
        im_height (int): Height of the image.
    Returns:
        PIL.Image: The background image with random transformations applied.
    """
    img = cv2.imread(background_path)
    img = cv2.resize(img, (im_width, im_height))

    # Random brightness (0.8 to 1.2)
    brightness_factor = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # Random contrast (-30 to +30)
    contrast_shift = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=contrast_shift)

    # Random zoom (1.0 to 1.2)
    zoom_factor = random.uniform(1.0, 1.2)
    if zoom_factor > 1.0:
        new_w = int(im_width / zoom_factor)
        new_h = int(im_height / zoom_factor)
        center_x, center_y = im_width // 2, im_height // 2
        x1 = center_x - new_w // 2
        y1 = center_y - new_h // 2
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        img = cv2.resize(cropped, (im_width, im_height))

    # Optional blur
    if random.random() < 0.5:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Optional hue/saturation shift
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hue_shift = random.randint(-10, 10)
        sat_shift = random.randint(-20, 20)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_shift, 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def overlay_backgrounds(output_dir, im_width, im_height):
    """
    Overlay the background on the images.
    Args:
        output_dir (str): Directory where the images are saved.
        im_width (int): Width of the images.
        im_height (int): Height of the images.
    """
    for img_path in glob.glob(os.path.join(output_dir, 'images', '*.png')):
        background = get_random_background(BACKGROUND_PATH, im_width, im_height)

        img = Image.open(img_path)

        background.paste(img, mask=img.convert("RGBA"))
        background.save(img_path)


def post_process_annotations(output_dir):
    """
    Post-process the annotations (change the categories so that all versions of the tools are in the same category, for visualization).
    Args:
        output_dir (str): Directory where the annotations are saved.
    """
    with open(os.path.join(output_dir, 'coco_annotations.json'), 'r') as f:
        annotations = json.load(f)
        annotations['categories'] = [
            {
                "id": 1,
                "supercategory": "coco_annotations",
                "name": "NeedleHolder"
            },
            {
                "id": 2,
                "supercategory": "coco_annotations",
                "name": "Tweezers"
            }
        ]

    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tools_folder', default=TOOLS_DEFAULT_FOLDER, help="Path to the tools objects' folder.")
    parser.add_argument('--camera_params', default=CAMERA_PARAMS_DEFAULT, help="Camera intrinsics in json format")
    parser.add_argument('--output_dir', default=OUTPUT_DIR_DEFAULT, help="Path to where the final files, will be saved")
    parser.add_argument('--num_images', type=int, default=NUM_IMAGES, help="Number of images to generate")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    bproc.init()

    needle_holders, tweezers, right_hand, left_hand, light = load_objects(args.tools_folder, RIGHT_HAND_FILE)

    cam2world_matrix, im_width, im_height = load_camera_params(args.camera_params)

    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    # sample camera poses
    for i in range(args.num_images):
        bproc.utility.reset_keyframes()

        needle_holder, tweezer = set_frame_positions(needle_holders, tweezers, right_hand, left_hand, light)

        bproc.camera.add_camera_pose(cam2world_matrix)

        # render the scene and save the annotations
        data = bproc.renderer.render()
        bproc.writer.write_coco_annotations(
                os.path.join(args.output_dir),
                instance_segmaps=data["instance_segmaps"],
                instance_attribute_maps=data["instance_attribute_maps"],
                colors=data["colors"],
                mask_encoding_format="rle",
                append_to_existing_output=True
            )

        needle_holder.hide()
        tweezer.hide()

    overlay_backgrounds(args.output_dir, im_width, im_height)

    post_process_annotations(args.output_dir)


if __name__ == "__main__":
    # Uncomment the following lines to enable debugging
    # debugpy.listen(5678)
    # debugpy.wait_for_client()

    main()