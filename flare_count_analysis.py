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


def class_to_num(flare_class):
    if flare_class == "B":
        return 0
    elif flare_class == "C":
        return 1
    elif flare_class == "M":
        return 2
    else:
        return 3

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
    info_df = pd.read_csv("classifiers/2013_2014_flare_info.csv")
    drop = list(set(info_df.columns) - {"nar", "time_start", "time_end", "xray_class"})
    info_df.drop(drop, axis=1, inplace=True)
    info_df.dropna(inplace=True)
    info_df.reset_index(inplace=True)
    info_df.drop("index", axis=1, inplace=True)

    info_df["xray_class"] = info_df["xray_class"].apply(classify_flare)
    info_df["class_num"] = info_df["xray_class"].apply(class_to_num)
    for time_string in ["time_start", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)
    print(info_df)
    # exit(1)

    coincidences = ["all", "coincident", "noncoincident"]
    labels = ["B", "C", "M", "X"]
    colors = ["blue", "green", "orange", "red"]
    for coincidence in coincidences:
        if coincidence == "coincident":
            flare_df = info_df.loc[info_df["is_coincident"] == True]
        elif coincidence == "noncoincident":
            flare_df = info_df.loc[info_df["is_coincident"] == False]
        for label, color in zip(labels, colors):
            flare_df = info_df.loc[info_df["xray_class"] == label]

            values, counts = np.unique(flare_df["nar"], return_counts=True)
            value_counts = [(int(value), count) for value, count in
                            zip(values, counts)
                            if not pd.isna(value)]
            value_counts = sorted(value_counts,
                                  key=lambda value_count: value_count[1],
                                  reverse=True)
            values = [value for value, _ in value_counts]
            counts = [count for _, count in value_counts]
            plt.plot(values, counts, color=color, label=label)
        plt.xlabel("AR #")
        plt.ylabel("# of Flares")

        plt.tight_layout()
        plt.show()



    # for value, count in value_counts:
    #     print(value, count)
    #
    # nar = 12297
    # for label in labels:
    #     df = info_df.loc[info_df["xray_class"] == label]
    #     df = df.loc[df["nar"] == nar]
    #     plt.scatter(df["time_start"], df["class_num"], label=label)
    # plt.legend(loc="best")
    # plt.xticks(rotation="vertical")
    # plt.yticks(color="w")
    # plt.title(f"Flare Coincidence for AR {nar} (140 Flares)")
    # plt.tight_layout()
    # plt.show()


def generate_time_plot():
    new_index = [
        'Jan 2013',
        'Feb 2013',
        'Mar 2013',
        'Apr 2013',
        'May 2013',
        'Jun 2013',
        'Jul 2013',
        'Aug 2013',
        'Sep 2013',
        'Oct 2013',
        'Nov 2013',
        'Dec 2013',
        'Jan 2014',
        'Feb 2014',
        'Mar 2014',
        'Apr 2014',
        'May 2014',
        'Jun 2014',
        'Jul 2014',
        'Aug 2014',
        'Sep 2014',
        'Oct 2014',
        'Nov 2014',
        'Dec 2014'
    ]

    info_df = pd.read_csv("classifiers/2013_2014_flare_info.csv")
    info_df["xray_class"] = info_df["xray_class"].apply(classify_flare)
    info_df["class_num"] = info_df["xray_class"].apply(class_to_num)
    for time_string in ["time_start", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    colors = ["blue", "green", "orange", "red"]
    flare_labels = ["B", "C", "M", "X"]
    coin_flare_df = pd.DataFrame(columns=flare_labels)
    noncoin_flare_df = pd.DataFrame(columns=flare_labels)

    plt.figure(figsize=(25, 25))

    for year in [2013, 2014]:
        for month in range(1, 12 + 1):
            flares = info_df.loc[
                (info_df['time_start'].dt.year == year) &
                (info_df['time_start'].dt.month == month)]
            # coin_b = flares.loc[(flares["xray_class"] == "B") & (flares["is_coincident"] == True)].shape[0]
            # coin_c = flares.loc[(flares["xray_class"] == "C") & (flares["is_coincident"] == True)].shape[0]
            # coin_m = flares.loc[(flares["xray_class"] == "M") & (flares["is_coincident"] == True)].shape[0]
            # coin_x = flares.loc[(flares["xray_class"] == "X") & (flares["is_coincident"] == True)].shape[0]
            # coin_flare_counts = [coin_b, coin_c, coin_m, coin_x]
            # coin_flare_df.loc[len(coin_flare_df)] = coin_flare_counts

            noncoin_b = flares.loc[(flares["xray_class"] == "B") & (flares["is_coincident"] == False)].shape[0]
            noncoin_c = flares.loc[(flares["xray_class"] == "C") & (flares["is_coincident"] == False)].shape[0]
            noncoin_m = flares.loc[(flares["xray_class"] == "M") & (flares["is_coincident"] == False)].shape[0]
            noncoin_x = flares.loc[(flares["xray_class"] == "X") & (flares["is_coincident"] == False)].shape[0]
            noncoin_flare_counts = [noncoin_b, noncoin_c, noncoin_m, noncoin_x]
            noncoin_flare_df.loc[len(noncoin_flare_df)] = noncoin_flare_counts

    # coin_flare_df.index = new_index
    noncoin_flare_df.index = new_index
    # coin_flare_df.plot(kind="bar", stacked=True, color=colors)
    noncoin_flare_df.plot(kind="bar", stacked=True, color=colors)
    plt.title("Non-coincidental Flare Counts for Solar Cycle 24, 2013-2014")
    plt.xticks(rotation="vertical", ha="center")
    plt.ylabel("# of Flares")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main()
    generate_time_plot()