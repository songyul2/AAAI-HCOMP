import sys
from pathlib import Path

import pandas as pd

import train_ai
import utils


def fusion(row):
    """
    Use sensor fusion to merge two normal distributions

    https://demonstrations.wolfram.com/SensorFusionWithNormallyDistributedNoise/
    """
    # the closed form results are 0.004 away from the combined estimates from stan samples.
    weight = row["variance_machine"] / (row["variance_machine"] + row["variance_human"])
    return row["calibrated_human"] * weight + row["calibrated_machine"] * (1 - weight)


def add_variance(df):
    # machine variance for every angle with no constraints

    variance = (
        df.groupby(["angle", "shapeIndex"])
        .apply(utils._compute_variance, include_groups=False)
        .round(3)
    )
    df["variance_machine"] = df.apply(
        lambda row: variance[(row["angle"], row["shapeIndex"])], axis=1
    )


def merge_df(
    df_human,
    df_machine=pd.read_csv("data/machine-data/preprocessed-test.tsv", sep="\t"),
    same_angle=False,
):
    """
    Merge the training or test part of the human data and the machines estimates on the same images. return a df.
    all images for the human experiment are part of the test data for the ai
    same_angle: bool, default False. If True, merge the human and machine estimates on the same angle. If False, merge the human and machine estimates for different views of the same jar.
    """

    df_human["angleIndex"] = (
        df_human["angle"].round(2).rank(method="dense", ascending=True)
    )
    # angle is normalized differently for ai and every fold of human. round so that the floats will be the same. dense: like ‘min’, but rank always increases by 1 between groups. we know there are 5 angles. rank is 1-based. dense rank is 1, 2, 3, 4, 5. 1 is the smallest angle. 5 is the largest angle
    df_machine["angleIndex"] = (
        df_machine["angle"].round(2).rank(method="dense", ascending=True)
    )
    # if we put no constraint on var, what would the results be? add_variance groups by angle and shapeIndex. if the angles are not rounded, the groups will be different
    # add_variance(df_machine)
    if same_angle:
        # img has the view angle in the file name. merge the human and machine estimates on the same angle
        df = df_human.merge(
            df_machine,
            on=["img", "id", "shapeIndex"],
            how="left",
            suffixes=("_human", "_machine"),
        )
    else:
        df = df_human.merge(
            df_machine,
            on=["id", "shapeIndex"],
            how="left",
            suffixes=("_human", "_machine"),
        )

    assert (
        df.notna().to_numpy().all()
    ), "all images for the human experiment are part of the test data for the ai"
    # print(abs(df["trueNum_machine"] - df["trueNum_human"]).max())
    assert 0 == round(
        abs(df["trueNum_machine"] - df["trueNum_human"]).max(), 8
    ), "true counts for the same jar even from different angles should be the same"

    return df


def _predict_fold(df, posterior_dir):
    """
    Use calibration parameters from the training stage to predict ground truth.
    Save the stan results to posterior_dir.
    Also save the joined df together with predictions and variance to a csv under calibrated_file_path.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        posterior_dir (str): The directory with the stan parameters posterior.
    """
    bias_variance = {}
    bias_variance["human"] = utils.read_samples(posterior_dir)
    bias_variance["machine"] = utils.read_samples(Path("data/machine-data"))
    combined_df = pd.DataFrame()
    for shape in utils.shapes6:
        df_shape = (
            df[df["shapeIndex"].str.contains(shape)].copy().reset_index(drop=True)
        )
        for agent in ["human", "machine"]:
            bias, variance = bias_variance[agent][shape]

            # 5 bias values for the 5 angles indexed by angleIndex
            bias = bias.iloc[
                df_shape[f"angleIndex_{agent}"].astype(int) - 1
            ].reset_index(drop=True)
            df_shape[f"calibrated_{agent}"] = df_shape[f"estimate_{agent}"] - bias

            df_shape[f"variance_{agent}"] = variance.iloc[
                df_shape[f"angleIndex_{agent}"].astype(int) - 1
            ].reset_index(drop=True)

        combined_df = combined_df._append(df_shape, ignore_index=True)
    return combined_df


def train_test_fold(train=False):
    """
    Train and test the fold given by sys.argv[1].

    Args:
        train (bool, optional): Flag indicating whether to perform training. Defaults to False.

    Returns:
        None
    """

    if len(sys.argv) > 1:
        fold = sys.argv[1]
        if train:
            train_test_dir = "human-data"
            results_dir = Path(f"data/{Path(train_test_dir).name}{fold}")
            utils.preprocess(train_test_dir, fold, results_dir)
            train_ai.cmdstan(results_dir)

        # we have learned the parameters. now we can predict the ground truth
        combined_df = pd.DataFrame()
        for subset in ["test", "train"]:
            # for subset in ["test"]:
            df_human = pd.read_csv(
                f"data/human-data{fold}/preprocessed-{subset}.tsv", sep="\t"
            )
            posterior_dir = Path(f"data/human-data{fold}")
            posterior_dir.mkdir(parents=True, exist_ok=True)

            merged = merge_df(df_human, same_angle=False)
            df_subset = _predict_fold(merged, posterior_dir)
            df_subset["calibrated_combined"] = df_subset.apply(fusion, axis=1)
            df_subset.index = pd.Index([subset] * len(df_subset))
            combined_df = combined_df._append(df_subset, ignore_index=False)
        combined_df.to_csv(Path(posterior_dir / "calibrated.tsv"), sep="\t")
    else:
        print("Usage: python join.py <fold> to process a specific fold")


if __name__ == "__main__":
    # compute the closed form bias and variance using the training data or read them from stan results
    # utils.read_samples = utils.compute_bias_variance
    train_test_fold(train=1)
