from pathlib import Path


import numpy as np
import pandas as pd


import utils

np.set_printoptions(suppress=True)

# stan integer array ii	a[ii] is	a[ii[1]], …, a[ii[K]]


def plot_prior():
    """plot the prior on variance depending on the angle"""
    s = np.random.randn(3, 10000) / (10**0.5)
    angles = np.array(
        [-1.36837988, -0.67377121, 0.05241059, 0.71544614, 1.47320106]
    ).reshape(-1, 1)
    # angles = np.array(sorted(["90", "0", "66", "45", "22"]), dtype=int).reshape(-1, 1)
    Angles = np.block([angles**0, angles, angles**2])  # 5 * 3

    variance = np.exp(Angles @ s)  # 5 * N
    fig, axs = plt.subplots(1, 5, figsize=(16, 4), layout="constrained")
    for i in range(5):
        varianceiable = variance[i]
        variance_range = varianceiable.max() - varianceiable.min()
        probabilities, values = np.histogram(varianceiable, bins=int(1e2), density=True)
        # Plot probability distribution like in your example
        axs[i].plot(values[:-1], probabilities)
    # df = pd.DataFrame(dict(value=values[:-1], prob=probabilities))
    # df.plot.line(x='value', y='prob')
    fig.savefig("plots/prior")


def toy_dataset():
    """Generate synthetic data with quadratic variance for stan to recover"""
    poly = PolynomialFeatures(3)
    # normalize the data! Return random floats in the half-open interval [0.0, 1.0)
    X_raw = np.random.random([10000, 2]) - 0.5
    X = poly.fit_transform(X_raw)
    N = X.shape[0]
    # parameters to recover
    beta = (np.random.random(10) - 0.5) * 1e1
    s = (np.random.random([3]) - 0.5) * 1e0
    variance = np.exp(X[:, [0, 1, 3]] @ s)  # N x 1
    y = np.random.normal(loc=X @ beta, scale=variance, size=N)

    # test data
    X_raw = np.random.random([10, 2]) * 10
    X_tilde = poly.fit_transform(X_raw)
    N_tilde = X_tilde.shape[0]

    dict_data = {
        "N": X.shape[0],
        "P": X.shape[1],
        "y": y,
        "X": X,
        "X_tilde": X_tilde,
        "N_tilde": N_tilde,
    }


def cmdstan(results_dir):
    """
    Runs the CmdStan model and saves the optimized parameters to a file.

    Parameters:
    - results_dir (str): The directory where the results will be saved.

    Returns:
    None
    """
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file="generate.stan")
    df = pd.read_csv(results_dir / "preprocessed-train.tsv", sep="\t")

    for shape in utils.shapes6:
        df_shape = df[df["shapeIndex"].str.contains(shape)]
        # we only use the additional training data for the transformation
        y, Y = utils.polynomial(df_shape, transform_sam=False)

        angles = np.sort(np.unique(Y[:, 1]))  # unique angles from column 1 of Y
        dict_data = {
            "N": Y.shape[0],
            "P": Y.shape[1],
            "y": y,
            "Y": Y,
            "angles": angles,
        }

        mle = model.optimize(data=dict_data, show_console=0)
        mle.optimized_params_pd.describe().to_csv(
            results_dir / f"parameters-{shape}.tsv", sep="\t"
        )

        # fit = model.sample(data=dict_data)
        # df = fit.draws_pd()
        # df['variance[1]'].hist(bins=100, alpha=0.5, label=shape)  # posterior mean and mode are close



def transform_and_save(log=True, standardize=False):
    """
    after transforming the sam output,
    create f"transformed-{subset}" files under data/transformed that contain the transformed data in a new column 'transformed' for plotting

    """
    beta = {}

    train_dir = "density_train"
    test_dir = "density_test"
    for directory in [train_dir, test_dir]:
        path = Path(directory) / "img_data.csv"
        df = pd.read_csv(path)
        df["transformed"] = 0.0  # add a new column to the DataFrame

        if log:
            df[["estimate", "trueNum"]] = np.log(df[["estimate", "trueNum"]])
        else:
            # convert columns to float whenever possible. either 'b' (boolean), 'i' (integer), or 'u' (unsigned integer).
            df = df.apply(lambda x: x.astype("float64") if x.dtype.kind in "biu" else x)
        # if standardize:
        # if we standardize here, the mean and var for ground truth wont be saved, and we wont have anything to use for the human data.    # Standardize the data. x represents each group of values in the 'trueNum' and 'estimate' columns for each unique 'shapeIndex'.
        #     zscore = lambda x: (x - x.mean()) / x.std()
        #     # Apply the scaler to each group of values
        #     df[['trueNum', 'estimate']] = df.groupby('shapeIndex')[['trueNum', 'estimate']].transform(lambda x: zscore(x))
        for shape in utils.shapes6:
            df_shape = df[df["shapeIndex"].str.contains(shape)]
            if df_shape.empty:
                continue
            # we only use the additional training data for the transformation
            y, X = utils.polynomial(df_shape, transform_sam=True)

            if shape not in beta:
                beta[shape] = np.linalg.inv(X.T @ X) @ X.T @ y
                # least squares solution from the training data.

            # Set the "transformed" column to the matrix-vector product of Y and beta
            df.loc[df["shapeIndex"] == shape, "transformed"] = np.dot(X, beta[shape])
        subset = directory.split("_")[-1]
        # use the transformed data for plotting
        Path("data/transformed").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"data/transformed/transformed-{subset}.tsv", sep="\t")
        df["estimate"] = df["transformed"]
        Path(f"data/sam/density_{subset}").mkdir(parents=True, exist_ok=True)
        df.drop(columns="transformed").to_csv(
            f"data/sam/density_{subset}/img_data.tsv", index=False, sep="\t"
        )
        # put the transformed data in column 'estimate'


def calibrate_machine(log=False, standardize=False):
    """If we take the log of the counts, it happens before regression for the AI"""
    utils.preprocess(
        train_test_dir="data/sam",
        fold="",
        results_dir="data/machine-data",
        log=log,
        standardize=standardize,
    )
    cmdstan(Path("data/machine-data"))


if __name__ == "__main__":
    transform_and_save()
    # draw.plot_counts_machine(subset="train")

    calibrate_machine()
