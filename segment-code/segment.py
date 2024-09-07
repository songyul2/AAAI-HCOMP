from pathlib import Path
from helpers import run_one_image
import matplotlib.pyplot as plt
import numpy as np

segment = 0  # set to True to run the segmenter
subset = "train"
# Set a new default font size
plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["lines.markersize"] = 6

size_to_sl = {
    "Cylinder1": "Cylinder (S)",
    "Cylinder2": "Cylinder (L)",
    "Disk1": "Disk (S)",
    "Disk2": "Disk (L)",
    "Sphere1": "Sphere (S)",
    "Sphere2": "Sphere (L)",
}


def filter_images(data):
    """
    Filter image names in data and discard some of them to make the values close to a uniform distribution.

    All but the last (righthand-most) bin is half-open. In other words, if bins is:

    [1, 2, 3, 4]
    then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4.
    """
    true_numbers = data[:, -1].astype(int)

    image_names = set()  # all image names to keep. Different views have different names
    values, bins = np.histogram(true_numbers, bins=10)
    value_min = (
        values.min()
    )  # how many in the smallest bin. save it since values will change later
    if value_min > 5:
        # the smallest bin has more than 6 images. this is hardcoded for Cylinder1. we need to discard some more images. avoid having too many images for a certain shape
        value_min = 5

    inds = np.digitize(true_numbers, bins)

    # this doesnt match the last bin which inclues its right endpoint!
    #  (bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]]
    for i, row in enumerate(data):
        # every row has a new line character at the end;
        imgname, num_of_objects, cam_view, shape, true_num = row
        true_num = int(true_num)
        if true_num == 0:
            raise ValueError(f"no object in the jar! {imgname}")

        index = inds[i] - 1  # which bin is this in
        if index == len(values):
            # this is the right endpoint of the last bin.
            index -= 1
        if true_num in [min(true_numbers), max(true_numbers)]:
            # this is the endpoint of the last bin or the first. if they stay there the equally spaced bins will be at the same places
            image_names.add(imgname[:-1])
            continue

        # the number of objects in this bin is above the threshold
        if values[index] > value_min:
            values[index] -= 1  # discard 1 image
            continue

        # The last character in the image name is the view number which can change
        image_names.add(imgname[:-1])
    return image_names


def generate_counts(data, im_dir, output_dir):
    """
    estimate the number of objects for every image in data and write the results to another CSV
    """
    view_map = dict(zip("12345", ["90", "0", "66", "45", "22"]))
    results_path = Path(output_dir) / "img_data.csv"
    with results_path.open("a") as results_file:
        if results_path.stat().st_size == 0:  # (file size in bytes):
            results_file.write("img,id,shapeIndex,angle,estimate,trueNum\n")
            # output csv header
        for row in data:
            imgname, num_of_objects, cam_view, shape, true_num = row
            image_path = (Path(im_dir) / imgname).with_suffix(".png")
            estimate = run_one_image(mask_generator, image_path, output_dir, true_num)
            if estimate <= 0:
                raise ValueError(f"estimate is {estimate} for {image_path}")

            new_row = [
                imgname,
                imgname.split("_")[1],
                shape,
                view_map[cam_view[-1]],  # view numbers to angles
                f"{estimate}",
                true_num,
            ]

            results_file.write(",".join(new_row) + "\n")


def pick_view1(data):
    """pick all rows in data with view1"""
    filter_indices = np.where("view1" == data[:, 2])
    # Different views will have the same true number of objects. So we pick one to filter the images
    return data[filter_indices]


def load_data(csv_path, shape, skiprows=0):
    """only return the np array with a shape in str type. shape can be Cylinder1. skiprows is the number of rows to skip in the csv file but it's not important here since we use shape to filter the rows. and row 0 is the header line with no shape."""
    # the csv header line will be excluded because it doesnt contain shape!
    # jar csv format
    # imgname,num_of_objects,cam_view,shape,true_num
    # col 0 is the image names without the parent directory path or extension
    data = np.loadtxt(
        csv_path,
        dtype=str,
        delimiter=",",
        skiprows=skiprows,
    )
    # all row indices that contains a shape
    shape_indices = [i for i in range(data.shape[0]) if shape in data[i, -2]]

    return data[shape_indices]


def adjust_distribution(axs, im_dir, filter=True):
    """
    im_dir contains the input images and csv. plot the distribution. return filtered input csv data for all images under im_dir.

    if filter is true, we discard images to make sure that there are the same number of images in every range of counts. if filter is false we keep all images. After that the subset of images will be passed to the segmenter.
    """
    data_shapes = []
    axs[0, -1].set_ylabel("Before filtering", rotation=270, labelpad=20)
    axs[1, -1].set_ylabel("After filtering", rotation=270, labelpad=20)
    axs[0, -1].yaxis.set_label_position("right")
    axs[1, -1].yaxis.set_label_position("right")

    for i, shape in enumerate(
        ["Cylinder1", "Cylinder2", "Disk1", "Disk2", "Sphere1", "Sphere2"]
    ):
        csv_path = im_dir / "jarexp_img_data.csv"
        data = load_data(csv_path, shape)
        if data.shape[0] == 0:  # no rows in this csv file!
            continue
        data1 = pick_view1(data)  # all rows with view1

        true_numbers = data1[:, -1].astype(int)

        # print(f"{min(true_numbers)} and {max(true_numbers)} for {size_to_sl[shape]}, ")

        # # """For a certain number of objects generated how many of them will stay in the jar"""
        # axs[0, i].scatter(num_of_objects, true_num, s=16, alpha=1)
        # axs[0, i].set(xlabel="num_of_objects", ylabel="true_num", title=f"{shape}")
        axs[0, i].set_title(f"{size_to_sl[shape]}")
        values, bins, _ = axs[0, i].hist(true_numbers, edgecolor="black", bins=10)

        if filter:
            # Remove the min and max values
            min_index = np.argmin(true_numbers)
            max_index = np.argmax(true_numbers)
            data1 = np.delete(data1, [min_index, max_index], axis=0)

            image_names = filter_images(data1)

            filter_indices = [
                i for i in range(data.shape[0]) if data[i, 0][:-1] in image_names
            ]
            # all views in data filtered to include only some of the images
            data = data[filter_indices]
            true_numbers = pick_view1(data)[:, -1].astype(int)
            values, bins, _ = axs[1, i].hist(true_numbers, edgecolor="black", bins=10)

            print(f"{data.shape[0]} for {size_to_sl[shape]}, ")

        data_shapes.append(data)

    return np.vstack(data_shapes)


if segment:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # use sam hq. change this to where the pretrained model is stored.
    sam_checkpoint = "../../sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # the modified library always creates an object on cpu now. this is a different object, not the sam model
    sam.to(device="cuda:1")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )


output_dir = Path(f"density_{subset}")
output_dir.mkdir(parents=True, exist_ok=True)
if "train" == subset:
    dir_list = [
        Path(f"../render_path_{shape}") for shape in ["Cylinder", "Disk", "Sphere"]
    ]  # dir that contains the  images and csv
else:
    dir_list = [Path("../jarstudy")]

fig, axs = plt.subplots(2, 6, figsize=[16, 9], layout="constrained")
fig.supxlabel("Number of objects in the jar")
fig.supylabel("Number of jars")
for im_dir in dir_list[:]:
    # filter the input images for the training data
    data = adjust_distribution(axs, im_dir, filter=("train" == subset))

    if segment:
        generate_counts(data, im_dir, output_dir)  # SLOW! 0.3 min per image


fig.savefig(f"plots/distribution-{output_dir.stem}")
