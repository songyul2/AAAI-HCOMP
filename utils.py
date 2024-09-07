from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

shapes6 = ["Cylinder1", "Cylinder2", "Disk1", "Disk2", "Sphere1", "Sphere2"]
shapes3 = shapes6[0::2]  # size 1 or 2 only
print(f"{shapes3=}")


def compute_bias_variance(csv_dir):
    """
    Compute the bias or variance for every angle with no constraints.

    Parameters:
    - csv_dir (str): The directory path where the CSV files are located.
    - shape (str): The shape index to filter the data.
    - variance (bool): Whether to compute the bias or variance.

    Returns:
    - pandas.DataFrame: A DataFrame containing the computed bias or variance for each angle.
    """
    df_train = pd.read_csv(csv_dir / "preprocessed-train.tsv", sep="\t")
    results = {}
    for shape in shapes6:
        df = df_train[df_train["shapeIndex"].str.contains(shape)].copy()
        # Creates bias and variance columns
        # The transform method returns an object that is indexed the same (same size) as the one being grouped. https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html
        # negative residual
        df["temp"] = df["estimate"] - df["trueNum"]
        grouped = df.groupby("angle")
        bias = grouped["temp"].mean()
        df["bias"] = grouped["temp"].transform("mean")
        df["calibrated"] = df["estimate"] - df["bias"]
        # squared residual. not residual sum of squares (RSS)
        df["temp"] = (df["calibrated"] - df["trueNum"]) ** 2
        grouped = df.groupby("angle")
        variance = grouped["temp"].mean()

        results[shape] = bias, variance
    return results


def human_bias_variance(shape):
    """
    Calculate the average human bias over 5 folds.

    Parameters:
    - shape: The shape of the data.

    Returns:
    - unconstrained_bias: The average human bias over 5 folds.
    """
    unconstrained_bias = pd.Series([0, 0, 0, 0, 0])
    unconstrained_variance = pd.Series([0, 0, 0, 0, 0])
    # accumulate the bias or variance over the 5 folds
    for fold in range(5):
        csv_dir = Path(f"data/human-data{fold}")
        bias, variance = compute_bias_variance(csv_dir)[shape]
        unconstrained_bias += bias.values
        unconstrained_variance += variance.values
    return unconstrained_bias / 5, unconstrained_variance / 5


def _transform_jarstudy4(df):
    """
    https://pandas.pydata.org/pandas-docs/dev/user_guide/indexing.html#indexing-view-versus-copy
    https://www.dataquest.io/blog/settingwithcopywarning/
    Remove outliers based off of the estimates for the same image.

    input df has columns:  estimate         id img  truenum  viewindex         PROLIFIC_PID
    """
    from scipy import stats
    group_estimate = df.groupby("img")["estimate"]
    group_z = group_estimate.transform(stats.zscore).abs()
    # keep only the rows where the z score is less than 1. 4932 / 6211 rows. Filtering too much human data breaks the symmetry of the experiment
    df = df.loc[group_z < 3].copy()
    # df = df.loc[group_z < 1] # this will raise SettingWithCopyWarning
    # We can use the drop parameter to avoid the old index being added as a column:
    df.reset_index(drop=True, inplace=True)
    df["img"] = df["img"].apply(lambda x: x.split(".")[0])  # remove .png
    # img names Cylinder2_V535C6Z59_view1 are separated by _
    df["shapeIndex"] = df["img"].apply(lambda x: x.split("_")[0])
    # for human data we have view indices 1 - 5. view num to angles. 90 is the top view. viewindex starts from 1, which maps to 90.
    mapping = ["", "90", "0", "66", "45", "22"]
    # this column will be dropped later in the caller function
    df["viewIndex"] = df["img"].apply(lambda x: x.split("_")[-1][-1])
    df["angle"] = df["viewIndex"].map(lambda x: mapping[int(x)])

    # rename 'truenum'!
    df = df.rename(columns={"truenum": "trueNum"})
    return df


def read_jarstudy4(dir=Path("human-data")):
    """read the data from the human experiments, split and save as train and test. Different folds are under the same directory with different names"""
    #  estimate         id                            img  truenum  viewindex         PROLIFIC_PID
    df = pd.read_csv(dir / "jarstudy4.csv", usecols=[0, 1, 2, 6, 7, 9])

    df = _transform_jarstudy4(df)

    # reorder columns and remove PROLIFIC_PID
    column_order = ["img", "id", "shapeIndex", "angle", "estimate", "trueNum"]
    df = df[column_order]
    df.to_csv(dir / "img_data.tsv", header=1, index=False, sep="\t")
    # 5 fold cross validation. one jar has 5 images. https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
    # gkf = GroupKFold(n_splits=5)
    # shuffle bool, default=False Whether to shuffle each class’s samples before splitting into batches.
    gkf = KFold(n_splits=5, shuffle=True)

    # save training and test for all folds
    train_dir = dir / "density_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir = dir / "density_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    fold = 0
    for train_index, test_index in gkf.split(df):
        # index, default True. Write row names (index).
        for df_subset, dir in zip(
            [df.loc[test_index], df.loc[train_index]], [test_dir, train_dir]
        ):
            df_subset.to_csv(
                dir / f"img_data{fold}.tsv", header=True, index=False, sep="\t"
            )

        fold += 1


def _load_filter_shape(path, shape):
    """
    Load a dataframe from a CSV file and filter rows based on a given shape.

    Parameters:
    path (str): The path to the CSV file.
    shape (str): The shape to filter rows by.

    Returns:
    pandas.DataFrame: The filtered dataframe where rows correspond to the specified shape.
    """
    df = pd.read_csv(path, sep="\t")
    return df[df["shapeIndex"].str.contains(shape)]


def _preprocess(train_data, test_data, log, standardize):
    """
    this is only used in one place!
    Standardize the angles. Optionally take the log of the counts or standardize the estimate and trueNum columns.
    """
    for df in [train_data, test_data]:
        # Convert numeric columns to float
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].astype(float)

        # # Standardize the angles.
        # angles = [90, 0, 66, 45, 22]
        # standardized_numbers = (np.array(angles) - np.mean(angles)) / np.std(angles)
        # mapping = {
        #     key: value for key, value in zip(angles, standardized_numbers.flatten())
        # }
        # Used for substituting each value in a Series with another value, that may be derived from a function, a dict or a Series.
        # df.iloc[:, 3] = df.iloc[:, 3].map(mapping)
        # Convert angles from degrees to radians. vectorized operations are faster than loops used implicitly by apply
        df["angle"] = np.deg2rad(df["angle"])

        if log:
            # Log of estimate and trueNum
            df.iloc[:, -2:] = np.log(df.iloc[:, -2:])

    if standardize:
        stats = {}
        # Standardize the estimate and trueNum columns for each unique shape. The index_col parameter of the pd.read_csv function is used to define the first (0th) column as index of the resulting DataFrame. Multiple columns can be set as index by passing a list of column numbers.
        stats["trueNum"] = pd.read_csv(
            "data/machine-data/trueNum_stats.tsv", sep="\t", index_col=0
        )
        stats["estimate"] = train_data.groupby("shapeIndex")["estimate"].agg(
            {"mean", "var"}
        )

        # Apply the same transformation to both training and test data
        for dataset in [train_data, test_data]:
            for shape_index in shapes6:
                mask = dataset["shapeIndex"] == shape_index

                # Standardize the estimate and trueNum columns
                for column in ["estimate", "trueNum"]:
                    mean = stats[column].loc[shape_index, "mean"]
                    variance = stats[column].loc[shape_index, "var"]
                    dataset.loc[mask, column] = (
                        dataset.loc[mask, column] - mean
                    ) / variance**0.5

    return train_data, test_data


def preprocess(
    train_test_dir, fold, results_dir, log=True, standardize=False, **kwargs
):
    """
    Preprocesses the training and test data under (train_test_dir / f"density_train/img_data{fold}.csv") and saves the results to results_dir / f"preprocessed-{subset}.tsv"

    Args:
        train_test_dir (str): The directory path where the training and test data is located.
        fold (str): The fold number. when fold is empty, we have the ai data.
        results_dir (str): The directory path where the preprocessed data will be saved.
        **kwargs: Additional keyword arguments, unused in this function.

    Returns:
        None
    """
    results_dir = Path(results_dir)
    train_test_dir = Path(train_test_dir)

    train_data = pd.read_csv(
        train_test_dir / f"density_train/img_data{fold}.tsv", sep="\t"
    )
    if standardize and "" == fold:
        # if we dont take the log, then we standardize. mean and variance for the ground truth must be shared between machine and human data. but for the estimates, mean and variance can be different. Calculate the mean and standard deviation for each shape in the training data
        trueNum_stats = train_data.groupby("shapeIndex")["trueNum"].agg({"mean", "var"})
        Path("data/machine-data").mkdir(parents=True, exist_ok=True)
        trueNum_stats.to_csv("data/machine-data/trueNum_stats.tsv", sep="\t")

    test_data = pd.read_csv(
        train_test_dir / f"density_test/img_data{fold}.tsv", sep="\t"
    )

    csv_data = _preprocess(train_data, test_data, log=log, standardize=standardize)
    results_dir.mkdir(parents=True, exist_ok=True)
    for i, subset in enumerate(["train", "test"]):
        results_file = results_dir / f"preprocessed-{subset}.tsv"
        csv_data[i].to_csv(results_file, index=False, sep="\t")


def read_samples(results_dir):
    """
    Read samples from the specified results directory.

    Parameters:
    - results_dir (str): The path to the directory containing the results.

    Returns:
    - dict: A dictionary containing the results for each shape.

    """

    def execute(results_dir, shape):
        summary = pd.read_csv(
            results_dir / f"parameters-{shape}.tsv", index_col=0, sep="\t"
        )
        # index_col=0 tells pandas to use the first column as the row index. otherwise the first column is treated as data. the first column contains index mean variance etc that we want to use as the row index
        # all columns that contain beta or variance parameters
        parameters = summary.filter(regex="bias|variance", axis="columns").loc["mean"]

        # last 5 columns are the variance parameters. in front of those are the bias parameters for 5 angles
        return parameters.iloc[-10:-5], parameters.iloc[-5:]

    results = {}
    for shape in shapes6:
        results[shape] = execute(results_dir, shape)
    return results


def polynomial(csv_data, transform_sam: bool):
    """
    Generate polynomial features for the given shape.

    Args:
        shape (str): The shape to generate polynomial features for.
        path (Path): The path to the CSV file containing the data.
        transform_sam (bool): Whether to transform the prediction or the sam output.

    Returns:
        tuple: If `transform_sam` is True, returns a tuple containing the estimate array and the linear array.
                If `transform_sam` is False, returns a tuple containing the ground truth array and the polynomial features array.
                       If there is no data for the given shape, returns None. this is not a tuple

    print(poly.get_feature_names_out())
    x0 is the angle and x1 is the estimate.
    with 2 varianceiables we have ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']

    with degree 3 we have 10 coefficients
    ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2', 'x0^3', 'x0^2 x1',
       'x0 x1^2', 'x1^3']
    or with 3 varianceiables we have 20 coefficients
    ['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2',
       'x2^2', 'x0^3', 'x0^2 x1', 'x0^2 x2', 'x0 x1^2', 'x0 x1 x2',
       'x0 x2^2', 'x1^3', 'x1^2 x2', 'x1 x2^2', 'x2^3']
    """
    y_true = csv_data["trueNum"]
    if not transform_sam:
        # combine the angle and trueNum columns into a single array
        return np.array(csv_data["estimate"]), np.column_stack(
            (np.ones_like(y_true), csv_data["angle"], y_true)
        )
    else:
        # transform the angle and segmenter output columns into an np array
        poly = PolynomialFeatures(degree=2)
        poly_features = poly.fit_transform(csv_data[["angle", "estimate"]])
        # standardize all features except the first column for np.linalg.inv(X.T @ X). this leads to much bigger ai errors for spheres 0 degree
        # poly_features[:, 1:] = StandardScaler().fit_transform(poly_features[:, 1:])
        # y_true = StandardScaler().fit_transform(y_true.values.reshape(-1, 1))
        return y_true, poly_features


def extract_predictions(df_calibrated):
    """return a data fram with calibrated human estimates, calibrated machine estimates, their variance, combined predictions, and naive averages.

    df_calibrated is the summary of the stan samples
    """
    df_shape = pd.DataFrame()
    # calibrated_human is the calibrated human estimates. rhs calibrated_human.1      5.164508. If the DataFrame index doesn't exist in the series, it will assign NaN.
    for entity in ["human", "machine"]:
        df_shape[f"calibrated_{entity}"] = (
            df_calibrated.filter(regex=f"calibrated_{entity}").loc["mean"].to_numpy()
        )
        # the variance of the predictions is slightly different than variance from the data generating process
        df_shape[f"variance_{entity}"] = (
            df_calibrated.filter(regex=f"calibrated_{entity}")
            .loc["variance"]
            .to_numpy()
        )

    df_shape["combined"] = (
        df_calibrated.filter(regex="combined").loc[f"{'mean'}"].to_numpy()
    )

    return df_shape


if __name__ == "__main__":
    read_jarstudy4()
