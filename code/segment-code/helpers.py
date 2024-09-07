from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


"""## Automatic mask generation options
def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        

There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. Here is an example configuration that samples more masks:


mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

masks2 = mask_generator_2.generate(image)

print(len(masks2))

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks2)
plt.axis("off")
plt.show()
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""# Automatically generating object masks with SAM

Since SAM can efficiently process prompts, masks for the entire image can be generated by sampling a large number of prompts over an image. This method was used to generate the dataset SA-1B. 

The class `SamAutomaticMaskGenerator` implements this capability. It works by sampling single-point input prompts in a grid over the image, from each of which SAM can predict multiple masks. Then, masks are filtered for quality and deduplicated using non-maximal suppression. Additional options allow for further improvement of mask quality and quantity, such as running prediction on multiple crops of the image or postprocessing masks to remove small disconnected regions and holes.
"""

"""## Automatic mask generation

To run automatic mask generation, provide a SAM model to the `SamAutomaticMaskGenerator` class. Set the path below to the SAM checkpoint. Running on CUDA and with the default model is recommended.
To generate masks, just run `generate` on an image."""


def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    # ax = plt.gca()
    # If you create two subplots, the subplot that is created last is the current one.
    ax.set_autoscale_on(False)
    ax.axis("off")

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def run_one_image(mask_generator, image_path: Path, output_dir: str, true_num: str):
    """imgname is the name of the input image in the csv without extension. true_num is a string representing the true number of objects in the image. used in the plot's title"""
    import cv2
    # If the original path doesn’t have a suffix, the new suffix is appended instead.
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    """Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
    * `segmentation` : the mask
    * `area` : the area of the mask in pixels
    * `bbox` : the boundary box of the mask in XYWH format
    * `predicted_iou` : the model's own prediction for the quality of the mask
    * `point_coords` : the sampled input point that generated this mask
    * `stability_score` : an additional measure of mask quality
    * `crop_box` : the crop of the image used to generate this mask in XYWH format
    """

    """Show all the masks overlayed on the image."""

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), layout="tight")
    axs[0].axis("off")
    # axs1 axis off is in the function call
    axs[0].imshow(image)
    axs[1].imshow(np.ones_like(image) * 128)  # uniform background
    show_anns(masks, axs[1])
    fig.suptitle(f"Ground truth: {true_num} Number of segmentation masks: {len(masks)}", fontsize=20)
    # # filter masks based on areas
    # masks_filtered = distribution(masks, axs)
    # axs[0, 1].imshow(np.ones_like(image) * 128)  # uniform background
    # show_anns(masks_filtered, axs[0, 1])
    # axs[0, 1].set(
    #     title=f"true_num {true_num} masks_filtered {len(masks_filtered)}",
    # )

    fig.savefig(Path(output_dir) / f"{image_path.stem}")
    plt.close()
    return len(masks)


def distribution(masks, axs):
    """given a list of masks plot the distribution of the area of each. use a threshold to filter out the smallest areas and return filtered masks."""
    ax = axs[1, 0]
    areas = [mask["area"] for mask in masks]
    areas.sort()  # ascending order
    areas = areas[:-9]  # The biggest areas correspond to the background
    values, bins, _ = ax.hist(areas)
    ax.set(xlabel="area of segments", ylabel="number of segments")
    threshold = bins[1]
    masks = [mask for mask in masks if mask["area"] > threshold]
    # remove all masks with area too small. their area will fall into the small bins
    # axs.set(title=f"{values} {bins}")
    # print(f"{values} {bins}")
    return masks

