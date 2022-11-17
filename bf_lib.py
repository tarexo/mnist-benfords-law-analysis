import cv2 as cv
import numpy as np
import pandas as pd
import numpy.typing as npt
import plotly.graph_objects as go

from tqdm import tqdm
from math import log10, floor
from statics import FSD_FAST, FSD_SLOW, BASE_10, QUANTIZATION_TABLE, BROWSER_PLOTS


def get_dct_array(image_list: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the DCT for each element in the list, flattens the result and returns a one-dimensional array.

    Args:
        image_list (npt.ArrayLike): A list of images

    Returns:
        npt.ArrayLike: A one-dimensional array of DCT values
    """
    dcts = np.array([cv.dct(image) for image in image_list])
    dcts = dcts.flatten()
    return dcts


def to_fsd(values: npt.ArrayLike, mode: int = FSD_FAST) -> npt.ArrayLike:
    """Replaces each value in values with its first significant digit.

    Args:
        values (npt.ArrayLike): An array of float values

    Returns:
        npt.ArrayLike: The first significant digits of values
    """
    values = values[values != 0]
    fsd = []
    if mode == FSD_SLOW:
        for value in tqdm(values):
            num = int(np.floor(abs(value * (10 ** -int(floor(log10(abs(value))))))))
            fsd.append(num)
    elif mode == FSD_FAST:
        n = np.abs(values * np.power(np.full(values.shape, 10.), -np.floor(np.log10(np.abs(values).astype("float64"))))).astype("int")
        fsd.extend(n)
    return np.array(fsd)
        

def count_fsds(fsds: npt.ArrayLike, base: int = BASE_10) -> npt.ArrayLike:
    """Counts up the occurence of each digit, depending on the base (1-9 for base 10).

    Args:
        fsds (npt.ArrayLike): An array of digits

    Returns:
        npt.ArrayLike: An array of the summed digits of length base - 1
    """
    count = []
    for i in range(1,base):
        count.append(np.count_nonzero(fsds == i))
    return np.array(count)


def benfords_law() -> npt.ArrayLike:
    """Create the ground truth distribution according to benfords law for base10 digits.

    Returns:
        npt.ArrayLike: The benfords law distribution for base10 digits
    """
    bf_law = []
    for i in range(1,10):
        bf_law.append(log10(1 + (1 / i)))
    return np.array(bf_law)


def mae_to_benfords_law(fsds: npt.ArrayLike) -> float:
    bf_law = benfords_law()
    err = 0
    for i in range(len(bf_law)):
        err += np.abs(fsds[i] - bf_law[i])
    return err / len(bf_law)


def kullback_leibler_divergence(fsds: npt.ArrayLike) -> float:
    bf_law = benfords_law()
    err = 0
    for i in range(len(bf_law)):
        err += fsds[i] * np.log(fsds[i] / bf_law[i])
    return err# / len(bf_law)


def jensen_shannon_divergence(fsds: npt.ArrayLike) -> float:
    bf_law = benfords_law()
    
    err = 0
    for i in range(len(bf_law)):
        err += fsds[i] * np.log(fsds[i] / bf_law[i])

    for i in range(len(bf_law)):
        err += bf_law[i] * np.log(bf_law[i] / fsds[i])
    return err #/ len(bf_law)


def plot_df_comparison(fsd_count_dist: npt.ArrayLike, base: int = BASE_10, title: str = "Measurements vs. Benfords Law"):
    """Plot Benfords law against measured first significant digits.

    Args:
        fsd_count_dist (npt.ArrayLike): Measured first significant digits
        base (int, optional): Number base. Defaults to BASE_10.
        title (str, optional): Title of the produced plot. Defaults to "Measurements vs. Benfords Law".
    """
    df = pd.DataFrame()
    df["digit"] = [i for i in range(1, base, 1)]
    df["MNIST FSD count"] = fsd_count_dist
    df["Benfords Law (ground truth)"] = benfords_law()

    mae = np.round(mae_to_benfords_law(fsd_count_dist), decimals=6)
    kld = np.round(kullback_leibler_divergence(fsd_count_dist), decimals=6)
    jsd = np.round(jensen_shannon_divergence(fsd_count_dist), decimals=6)
    title = title + "\n - Mean absolute error: " + str(mae) + "\n - Kullback-Leibler: " + str(kld) + "\n - Jensen-Shannon: " + str(jsd)

    fig = go.Figure()
    fig.add_bar(x=df["digit"], y=df["MNIST FSD count"], name="Measurements", hoverinfo="y")
    fig.add_scatter(x=df["digit"], y=df["Benfords Law (ground truth)"], name="Ground Truth", hoverinfo="y")
    fig.update_layout(title=title, xaxis_title="Digits", yaxis_title="Distribution")

    fig.show(renderer="browser")


def img_to_blocks(image: npt.ArrayLike) -> npt.ArrayLike:
    """Divides the image into non-overlapping 8x8 blocks and returns them.

    Args:
        image (npt.ArrayLike): The image, from which the blocks should be extracted

    Returns:
        npt.ArrayLike: 8x8 blocks of the image
    """
    if len(image.shape) > 2:
        image = image.reshape(28,28)
    num_blocks = int(image.shape[0] / 8)
    blocks = []
    for row in range(0, (num_blocks) * 8, 8):
        for col in range(0, (num_blocks) * 8, 8):
            block = image[row:row+8, col:col+8]
            blocks.append(block)
    return np.array(blocks)


def quantize_block(block: npt.ArrayLike, q_table: npt.ArrayLike = QUANTIZATION_TABLE) -> npt.ArrayLike:
    """Takes a DC transformed 8x8 block and performs a quantization.

    Args:
        block (npt.ArrayLike): The 8x8 block (a 2d array)
        q_table (int, optional): The quantization table. Defaults to QUANTIZATION_TABLE.

    Returns:
        npt.ArrayLike: The quantized 8x8 block (a 2d array)
    """
    return np.abs(np.round(block / q_table))