import os

OUTPUT_DIR_DEFAULT = "synthetic_data"


def make_annotations_vis(n):
    """
    run the blenderproc vis command to display segmentation labels and save them
    Args:
        n (int): The number of images to visualize.
    """
    for i in range(n):
        os.system(f"blenderproc vis coco -i {i} -b {OUTPUT_DIR_DEFAULT} -s")


if __name__ == "__main__":
    make_annotations_vis(10)