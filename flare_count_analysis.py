import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib

import matplotlib
from astropy.io import fits
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import seaborn as sns
import sunpy.map
import sunpy.io
import datetime as dt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import csv
import plotly.express as px
from scipy.stats import chisquare

FLARE_PROPERTIES = [
    'ABSNJZH',
    'AREA_ACR',
    'MEANGAM',
    'MEANGBH',
    'MEANGBT',
    'MEANGBZ',
    'MEANJZD',
    'MEANJZH',
    'MEANPOT',
    'MEANSHR',
    'R_VALUE',
    'SAVNCPP',
    'SHRGT45',
    'TOTPOT',
    'TOTUSJH',
    'TOTUSJZ',
    'USFLUX',
]

to_drop = [
    'MEANGBZ',
    'MEANGBH',
    'TOTUSJZ',
    'TOTUSJH',
    'SAVNCPP',
    'MEANPOT',
    'TOTPOT',
    'MEANSHR',
    'SHRGT45',
    'AREA_ACR'
]

UNIQUE_PROPERTIES = list(set(FLARE_PROPERTIES) - set(to_drop))
UNIQUE_PROPERTIES = sorted(UNIQUE_PROPERTIES)

CLASS_LABELS = ["B", "C", "M", "X"]


def parse_tai_string(tstr, datetime=True):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    if datetime:
        return dt_obj(year, month, day, hour, minute)
    else:
        return year, month, day, hour, minute


def classify_flare(magnitude):
    if "B" in magnitude:
        return "B"
    elif "C" in magnitude:
        return "C"
    elif "M" in magnitude:
        return "M"
    else:
        return "X"


def floor_minute(time, cadence=12):
    return time - datetime.timedelta(minutes=time.minute % cadence)


to_drop = [
    'MEANGBZ',
    'MEANGBH',
    'TOTUSJZ',
    'TOTUSJH',
    'SAVNCPP',
    'MEANPOT',
    'TOTPOT',
    'MEANSHR',
    'SHRGT45',
    'AREA_ACR'
]

def main():
    info_df = pd.read_csv("all_flares.txt")
    drop = list(set(info_df.columns) - {"nar", "time_start", "time_end", "xray_class"})
    info_df.drop(drop, axis=1, inplace=True)
    info_df.dropna(inplace=True)
    info_df.reset_index(inplace=True)
    info_df.drop("index", axis=1, inplace=True)

    def class_to_num(flare_class):
        if flare_class == "B":
            return 0
        elif flare_class == "C":
            return 1
        elif flare_class == "M":
            return 2
        else:
            return 3

    info_df["xray_class"] = info_df["xray_class"].apply(classify_flare)
    info_df["class_num"] = info_df["xray_class"].apply(class_to_num)
    for time_string in ["time_start", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)
    print(info_df)
    # exit(1)

    labels = ["B", "C", "M", "X"]

    values, counts = np.unique(info_df["nar"], return_counts=True)
    value_counts = [(int(value), count) for value, count in zip(values, counts)
                    if not pd.isna(value)]
    value_counts = sorted(value_counts, key=lambda value_count: value_count[1],
                          reverse=True)
    # values = [value for value, _ in value_counts]
    # counts = [count for _, count in value_counts]
    # plt.plot(values, counts)
    # plt.xlabel("AR #")
    # plt.ylabel("# of Flares")
    # plt.tight_layout()
    # plt.show()
    for value, count in value_counts:
        print(value, count)

    nar = 12297
    for label in labels:
        df = info_df.loc[info_df["xray_class"] == label]
        df = df.loc[df["nar"] == nar]
        plt.scatter(df["time_start"], df["class_num"], label=label)
    plt.legend(loc="best")
    plt.xticks(rotation="vertical")
    plt.yticks(color="w")
    plt.title(f"Flare Coincidence for AR {nar} (140 Flares)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()