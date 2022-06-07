import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib

import matplotlib
from astropy.io import fits
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import seaborn as sns
import sunpy.map
import sunpy.io
from IPython.display import Image
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import csv
import plotly.express as px

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


def main():
    abc_properties_df = pd.read_csv("Data_ABC.csv")
    mx_properties_df = pd.read_csv("Data_MX.csv")

    info_df = pd.read_csv("all_flares.txt")
    info_df.drop(["hec_id", "lat_hg", "long_hg", "long_carr", "optical_class"], axis=1, inplace=True)

    # Convert time strings to datetime objects for cleaned info data.
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)
    properties_df = pd.concat([abc_properties_df, mx_properties_df])

    # Label flares by B, C, M, and X.
    info_df["xray_class"] = \
        info_df["xray_class"].apply(classify_flare)

    def info_to_data(info, data):
        df = pd.DataFrame()
        for index, row in info.iterrows():
            # if flare_coincidences_mix[index] <= 0:
            # if flare_coincidences_mix[index] > 0:
            #     continue
            print(row)
            timestamp = row["time_start"]
            df_sort = data.iloc[
                (data['T_REC'] - timestamp).abs().argsort()[:1]]
            df_sort.insert(0, "xray_class", row["xray_class"])
            # df_sort["xray_class"] = row["xray_class"]
            df = pd.concat([df, df_sort])
            print(f"{index}/{info.shape[0]}", df)
            if index == 30:
                break
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        return df

    df = info_to_data(info_df, properties_df)
    df.drop(["xray_class", "T_REC"], axis=1, inplace=True)
    cm = df.corr()
    print(cm)

if __name__ == "__main__":
    main()