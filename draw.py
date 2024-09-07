import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

import utils

# Set a new default font size
plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
# plt.rcParams["axes.labelsize"] = 20
plt.rcParams["lines.markersize"] = 6
angles5 = sorted(["90", "0", "66", "45", "22"])
# dont use size 1 and 2. use S and L in plots
size_to_sl = {
    "Cylinder1": "Cylinder (S)",
    "Cylinder2": "Cylinder (L)",
    "Disk1": "Disk (S)",
    "Disk2": "Disk (L)",
    "Sphere1": "Sphere (S)",
    "Sphere2": "Sphere (L)",
}


def _compute_mse(df):
    """
    Compute the error for different columns in the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the necessary columns.

    Returns:
    - result (pandas.DataFrame): A DataFrame containing the computed mse values for each column.

    """
    # these 2 angles can be different
    df = df[df["angle_human"] == df["angle_machine"]].copy()

    # the average of the human and machine serves as a naive baseline
    df = df.assign(baseline=(df["calibrated_human"] + df["calibrated_machine"]) / 2)
    result = pd.DataFrame()
    # sort: bool, default True. Sort group keys. angle is normalized float
    grouped = df.groupby("angle_human")

    for j, name_group in enumerate(grouped):
        df_angle = name_group[1]
        for col in [
            "estimate_human",
            "estimate_machine",
            "calibrated_human",
            "calibrated_machine",
            "calibrated_combined",
            "baseline",
        ]:
            if col in df_angle.columns:
                try:
                    y_mse = mean_squared_error(df_angle["trueNum_human"], df_angle[col])
                    result.loc[j, col] = y_mse
                except ValueError:
                    breakpoint()
    return result


def _cross_validate(df_dict, shape, compute_metric=_compute_mse):
    fold = 0
    df = df_dict[fold]
    df_shape = df[df["shapeIndex"].str.contains(shape)]
    # accumulate the sum of errors over folds for each angle. assume that the first fold exists!
    result = compute_metric(df_shape)

    while fold < 5:
        fold += 1
        if (fold) not in df_dict:
            break
        df = df_dict[fold]
        df_shape = df[df["shapeIndex"].str.contains(shape)]
        error = compute_metric(df_shape)
        result += error

    return result / fold


def plot_axs_23(ylabel):
    """
    A function that creates a 3 x 2 grid of subplots and decorates them with titles, labels, and legends.

    Parameters:
    - ylabel (str): The label for the y-axis of the subplots.

    Returns:
    - function: A decorator that takes in plot_ax and plots the data and returns the figure.

    Example usage:
    @plot_axs_23("Y Label")
    def my_plot(ax, shape, **kwargs):
        # plot the data on the given axis
        pass

    decorated_plot = my_plot(**kwargs)
    fig = decorated_plot()
    """

    def decorate(plot_ax):
        def run(*args, **kwargs):
            fig, axs = plt.subplots(
                3, 2, layout="constrained", sharex="col", figsize=(10, 9)
            )
            for s, shape in enumerate(utils.shapes6):
                j = s % 2  # row index. 0 or 1.
                i = s // 2  # column index. 0, 1, 2.
                axs[i, j].set_title(f"{size_to_sl[shape]}")
                plot_ax(axs[i, j], shape, *args, **kwargs)

            handles, labels = axs[0, 0].get_legend_handles_labels()
            # if there are any labels, use the label in one subplot as the fig legend.
            # Sometimes it makes more sense to place a legend relative to the (sub)figure rather than individual Axes. By using constrained layout and specifying "outside" at the beginning of the loc keyword argument, the legend is drawn outside the Axes on the (sub)figure.
            if handles and labels:
                fig.legend(handles, labels, loc="outside right")
            if ylabel:
                # if ylabel is nonempty, then we assume angle is the xlabel
                fig.supylabel(ylabel)
                fig.supxlabel("Angle $a$", x=0.42)    # this wont be centered in the figure if the legend is present outside the axes
                # ax.transAxes	The coordinate system of the Axes; (0, 0) is bottom left of the axes, and (1, 1) is top right of the axes.
                # axs[1, 1].text(
                #     0.5,
                #     -0.2,
                #     "Angle $a$",
                #     fontsize=20,
                #     transform=axs[1, 1].transAxes,
                #     ha="center",
                # )

            return fig

        return run

    return decorate


def plot_same_angle(ax, shape, df_dict, compare_calibration):
    """plot the error for humans, machines and the combined predictions. average over five fold cross validation on the human estimates."""

    result = _cross_validate(df_dict, shape)
    if compare_calibration:
        # compare the results before and after calibration
        result = result[
            [
                "estimate_human",
                "calibrated_human",
                "estimate_machine",
                "calibrated_machine",
            ]
        ]
    else:
        result = result[
            ["calibrated_human", "calibrated_machine", "calibrated_combined"]
        ]

    ax.set(title=f"{size_to_sl[shape]}")
    column_label = {
        "estimate_human": "Human estimates",
        "estimate_machine": "AI estimates",
        "calibrated_human": "Human",
        "calibrated_machine": "AI",
        "calibrated_combined": "Combined",
        "baseline": "Average",
    }
    for col in result.columns:
        label = column_label.get(col, "")
        ax.plot(
            angles5,
            result[col],
            label=label,
            marker="o",
            markersize=8,
            linestyle="dashed",
        )


def _cache_data():
    # df_dict avoids reading the same csv file multiple times. we put the data frames for all folds in a dictionary. they are read only once. memory is cheap. we can afford to store the data frames in memory.
    df_dict = {}
    for fold in range(5):
        path = Path(f"data/human-data{fold}/calibrated.tsv")
        if path.exists():
            df_dict[(fold)] = pd.read_csv(path, sep="\t", index_col=0)

        else:
            print(f"missing {path}")
            break
    return df_dict


def mse_plot(subset="test", compare_calibration=False):
    """test is test data for humans"""
    if compare_calibration:
        plot_path = Path(f"plots/mse-calibrate-{subset}")
    else:
        plot_path = Path(f"plots/mse-combine-{subset}")
    df_dict = _cache_data()
    # TODO df_dict now contains the data for both training and testing
    plot_same_angle_partial = functools.partial(
        plot_same_angle, df_dict=df_dict, compare_calibration=compare_calibration
    )
    mse_fig = plot_axs_23("Mean squared error (MSE)")(plot_same_angle_partial)
    fig = mse_fig()
    fig.savefig(plot_path)


def _bias_variance_fig(ax, shape, variance=True):
    """
    Plot standard deviation values for human calibration, AI calibration, and AI transformation.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the data.
    - shape (str): The shape of the data.
    variance = 0  # plot bias or variance
    Returns:
    None
    """
    accumulator = pd.Series([0, 0, 0, 0, 0])
    # accumulate the bias or variance over the 5 folds
    for fold in range(5):
        csv_dir = Path(f"data/human-data{fold}")
        try:
            #  If the indices do not match, the operation will result in NaN for those indices.
            accumulator += utils.read_samples(csv_dir)[shape][variance].values

        except FileNotFoundError:
            print(f"missing {csv_dir}")
            accumulator /= fold
            break
    else:
        # the loop completed without a break
        accumulator /= 5

    ax.plot(
        angles5,
        utils.read_samples(Path("data/machine-data"))[shape][variance],
        label="AI",
        marker="o",
    )

    bias_var_machine = utils.compute_bias_variance(Path("data/machine-data"))[shape]
    # compute human bias for every fold then variance for every fold
    bias_var_human = utils.human_bias_variance(shape)

    ax.plot(
        angles5,
        bias_var_machine[variance],
        label="AI closed-form",
        linestyle="dashdot",
        marker="o",
    )
    ax.plot(
        angles5,
        accumulator,
        label="Human",
        marker="o",
    )
    ax.plot(
        angles5,
        bias_var_human[variance],
        label="Human closed-form",
        linestyle="dashdot",
        marker="o",
    )


def bias_variance_plot(mode):
    """
    Plot the bias or variance based on the given mode.

    Parameters:
    - mode (str): The mode indicating whether to plot bias or variance.

    Returns:
    None
    """

    variance = "variance" == mode
    fig = functools.partial(_bias_variance_fig, variance=variance)
    if variance:
        plot_axs_23("Variance $v$")(fig)().savefig("plots/variance-plot")
    else:
        plot_axs_23("Bias $b$")(fig)().savefig("plots/bias-plot")


def _average_mse_diff(mse_diff):
    # Calculate the average of mse_diff over the diagonal and the rest
    diagonal_avg = mse_diff.values.diagonal().mean()
    non_diagonal_avg = mse_diff.values[~np.eye(mse_diff.shape[0], dtype=bool)].mean()
    return diagonal_avg, non_diagonal_avg


def _ri(baseline: str, df_train_test: pd.DataFrame):
    """
    Calculate the improvement in mean squared error (MSE) over different baselines.

    Args:
        baseline (str): The type of baseline to calculate improvement for. Can be one of:
            - "pick": pick the best agent for every angle combination.
            - "average": Use the average of calibrated_human and calibrated_machine as the baseline.
            - "weighted": Use a weighted combination of calibrated_human and calibrated_machine as the baseline.
        df_train_test (pd.DataFrame): The DataFrame containing the training and test data.

    Returns:
        pd.Series: A Series containing the relative improvement in MSE for each unique combination of "angle_human" and "angle_machine".
    """
    df = df_train_test.loc["test"].copy()

    if "pick" == baseline:
        # the baseline predictions are implicit
        grouped = df.groupby(["angle_human", "angle_machine"])
        mse = {}
        for column in ["human", "machine"]:
            mse[f"mse_{column}"] = grouped.apply(
                lambda x: mean_squared_error(
                    x["trueNum_human"], x[f"calibrated_{column}"]
                ),
                # include_groups=False,
            )

        mse_baseline = pd.concat(mse.values(), axis=1).min(axis=1)

    else:
        # we can easily compute the baseline predictions
        if "average" == baseline:
            weight = 0.5
        elif "weighted" == baseline:
            df_train = df_train_test.loc["train"]
            weight = sum(
                (df_train["trueNum_human"] - df_train["calibrated_machine"])
                * (df_train["calibrated_human"] - df_train["calibrated_machine"])
            ) / sum(
                (df_train["calibrated_human"] - df_train["calibrated_machine"]) ** 2
            )

        df["baseline"] = df["calibrated_human"] * weight + df["calibrated_machine"] * (
            1 - weight
        )
        grouped = df.groupby(["angle_human", "angle_machine"])
        mse_baseline = grouped.apply(
            lambda x: mean_squared_error(x["trueNum_human"], x["baseline"]),
            # include_groups=False,
        )

    mse_combined = grouped.apply(
        lambda x: mean_squared_error(x["trueNum_human"], x["calibrated_combined"]),
        # include_groups=False,
    )
    relative_improvement = (mse_baseline - mse_combined) / mse_baseline
    return relative_improvement


def different_angle_plot(ax, shape, df_dict, baseline):
    compute_change = functools.partial(_ri, baseline)
    mse_diff = _cross_validate(df_dict, shape, compute_metric=compute_change).unstack()
    diagonal_avg, non_diagonal_avg = _average_mse_diff(mse_diff)

    fmt = ".2f"
    # latex table
    if 'average' == baseline:
        print(
        f"{size_to_sl[shape]}  {diagonal_avg * 100:{fmt}}  {non_diagonal_avg * 100:{fmt}}"
    )
    else:
        print(
        f"{diagonal_avg * 100:{fmt}}  {non_diagonal_avg * 100:{fmt}}"
    )
    sns.heatmap(
        mse_diff * 100,
        annot=True,
        ax=ax,
        cbar=False,
        cmap="icefire",
        center=0,
        square=True,
        xticklabels=angles5,
        yticklabels=angles5,
        fmt='.0f'
    )
    ax.set(title=f"{size_to_sl[shape]}", xlabel="", ylabel="")
    ax.invert_yaxis()


def different_angles_fig(baseline):
    """
    Generate a figure showing the relative improvement in mean squared error compared to the baseline
    for different shapes and angles.

    Parameters:
    - baseline: The baseline value to compare against.

    Returns:
    - fig: The generated figure object.
    """

    df_dict = _cache_data()
    plot_angle_partial = functools.partial(
        different_angle_plot, df_dict=df_dict, baseline=baseline
    )
    mse_fig = plot_axs_23("")(plot_angle_partial)
    fig = mse_fig()
    fig.supxlabel("AI angle")
    fig.supylabel("Human angle")
    return fig


def _plot_counts_shape(df, axs, shape, suffix, calibrate, transform):
    """
    Plot the counts for a specific shape and angles.

    Args:
        df: The data containing the counts.
        axs (list): The list of axes for the 5 angles.
        shape (str): The shape to plot the counts for.
        transform (bool): Whether we are plotting the transformed sam output
        suffix (str, optional): The source of the data if the df was joined. can be "_machine" or "_human" or empty
    """

    if df.shape[0] == 0:
        raise ValueError(f"no data for {shape}")

    axs[0].set(title=f"{shape[:-1]}")  # remove the size from the shape name

    unique_angles = sorted(set(df[f"angle{suffix}"]))
    assert 5 == len(
        unique_angles
    ), "we have 5 angles. if we find more this is a numerical issue."
    for i, angle in enumerate(unique_angles):
        # axs[i].set_aspect("equal", adjustable="datalim")

        # angle is on the right, opposite the y-axis label
        if "Sphere" in shape:
            axs[i].yaxis.set_label_position("right")
            axs[i].set_ylabel(f"${angles5[i]}^\circ$", rotation=270, labelpad=20)
        # floats are not exact. we need to use a tolerance
        angle_indices = [abs(a - angle) < 1e-6 for a in df[f"angle{suffix}"]]
        data_angle = df[angle_indices]
        if "human" not in suffix:
            # sam output
            if transform:
                axs[i].scatter(data_angle["estimate"], data_angle["transformed"], s=14)
            else:
                axs[i].scatter(data_angle["estimate"], data_angle["trueNum"], s=14)
        else:
            # many people were given the same image. we need to average the estimates
            true_num_groupby = data_angle.groupby(f"trueNum{suffix}")
            true_num = true_num_groupby.groups.keys()
            alpha = 1

            axs[i].scatter(
                true_num, true_num, label="Ground truth", marker="d", alpha=alpha
            )
            axs[i].scatter(
                true_num,
                true_num_groupby[f"estimate{suffix}"].mean(),
                label="Mean of raw estimates",
                alpha=alpha,
                marker="s",
            )

            if calibrate and f"calibrated{suffix}" in data_angle.columns:
                # the df has the calibrated estimates and their std. grougby sorts by the group keys
                calibrated = true_num_groupby[f"calibrated{suffix}"].mean()
                # std = true_num_groupby[f"variance{suffix}"] ** 0.5 .mean()
                axs[i].scatter(
                    true_num,
                    calibrated,
                    label="Mean of calibrated estimates",
                    alpha=alpha,
                )

                # # Plot bands instead of errorbars
                # axs[i].fill_between(
                #     true_num,
                #     calibrated - std,
                #     calibrated + std,
                #     alpha=1 / 16,
                #     label="Calibrated estimate +/- Standard deviation")


def plot_counts_all(csv_path: Path, calibrate: bool, transform: bool, suffix: str):
    """
    Plot counts for all shapes in a given CSV file.

    Args:
        csv_path (Path): The path to the CSV file.
        predicitions (bool): Whether to include predictions in the plot.
        suffix (str): _machine or _human in the joined data. if empty, the data is not joined.

    Returns:
        None
    """
    shapes = utils.shapes3
    # 'constrained': The constrained layout solver adjusts axes sizes to avoid overlapping axes decorations. Can handle complex plot layouts and colorbars, and is thus recommended.
    fig, axs = plt.subplots(
        5, len(shapes), figsize=[9, 9], layout="constrained", sharex="col"
    )
    df = pd.read_csv(csv_path, sep="\t")

    for i, shape in enumerate(shapes):
        _plot_counts_shape(
            df[df["shapeIndex"].str.contains(shape)],
            axs[:, i],
            shape,
            suffix=suffix,
            calibrate=calibrate,
            transform=transform,
        )

    # use the labels one subplot as the fig legend
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # 'outside upper right' will make space for the legend above the rest of the axes in the layout, and 'outside right upper' will make space on the right side of the layout. this approach will shift the axes to the left!
    # fig.legend(handles, labels, loc="outside upper center")
    # instead, use the ax in the center row and the last column to place the legend. bbox_to_anchor is the position of the legend. bbox_transform is the coordinate system of the bbox.
    # if there is a legend
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            bbox_transform=axs[2, -1].transAxes,
        )

    if "human" in suffix:
        # Individual parts of a path divided by slashes. The stem property of a pathlib.Path object gives the final path component, without its suffix. csv_path.parts[-2] can be human-data4
        fig_path = f"plots/{csv_path.parts[-2]}-{csv_path.stem}.png"
        fig.supxlabel("Ground truth")
    # machine estimate or transformed sam output
    else:
        stage = "after" if transform else "before"
        fig_path = f"plots/{csv_path.stem}-{stage}-size{utils.shapes3[0][-1]}.png"
        fig.supxlabel("Number of segmented objects $n_i$")
        if transform:
            fig.supylabel("AI estimate $y_{mi}$")
        else:
            fig.supylabel("Ground truth $y_i$")
    Path("plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    # 'tight': This option adjusts the bounding box of the saved figure to include all artist elements on the canvas (i.e., anything that's drawn on the plot).


def plot_counts_machine(subset="train", transform=True):
    """plot the transformed segmenter output and see if it is closer to the ground truth than the raw segmenter output."""
    results_path = Path(f"data/transformed/transformed-{subset}.tsv")
    plot_counts_all(
        results_path,
        transform=transform,
        suffix="",
        calibrate=False,
    )


def plot_counts_human(subset="train"):
    # the training data for human and machine is not the same. it unfair to compare the mse on the training data
    results_path = Path("data/human-data0/calibrated.tsv")
    plot_counts_all(
        results_path,
        transform=False,
        suffix="_human",
        calibrate=True,
    )


if __name__ == "__main__":
    # subset = "train"
    # plot_counts_machine(subset, transform=True)
    # plot_counts_machine(subset, transform=False)

    # mse_plot(subset='test', compare_calibration=False)
    # bias_variance_plot(mode="bias")
    # bias_variance_plot(mode="variance")
    for baseline in ["average", "weighted", "pick"]:
        different_angles_fig(baseline).savefig(f"plots/mse-diff-{baseline}")
