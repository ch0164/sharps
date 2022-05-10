import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib
from astropy.io import fits
from sklearn.decomposition import PCA
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import sunpy.map
import sunpy.io
from IPython.display import Image
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import csv

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
    # Choose which flares to plot.
    # ABC Flares
    abc_info_df = pd.read_csv("ABC_list.txt")
    abc_properties_df = pd.read_csv("Data_ABC.csv")
    # MX Flares
    mx_info_df = pd.read_csv("MX_list.txt")
    mx_properties_df = pd.read_csv("Data_MX.csv")

    # Convert time strings to datetime objects for cleaned info data.
    for time_string in ["time_start", "time_peak", "time_end"]:
        abc_info_df[time_string] = \
            abc_info_df[time_string].apply(parse_tai_string)
        mx_info_df[time_string] = \
            mx_info_df[time_string].apply(parse_tai_string)

    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    # Label flares by B, C, M, and X.
    abc_info_df["xray_class"] = \
        abc_info_df["xray_class"].apply(classify_flare)
    mx_info_df["xray_class"] = \
        mx_info_df["xray_class"].apply(classify_flare)

    # Get B and C class flares, round down their minutes.
    b_df = abc_info_df.loc[abc_info_df["xray_class"] == "B"]
    print("Class B Flares Shape: ", b_df.shape)
    c_df = abc_info_df.loc[abc_info_df["xray_class"] == "C"]
    print("Class C Flares Shape: ", c_df.shape)
    bc_info = pd.concat([b_df, c_df])
    bc_info["time_start"] = \
        bc_info["time_start"].apply(floor_minute)

    # Find the respective timestamp in the ABC data file.
    bc_data = pd.DataFrame()
    for index, row in bc_info.iterrows():
        timestamp = row["time_start"]
        bc_series = abc_properties_df.loc[
            abc_properties_df["T_REC"] == timestamp].head(1)
        # bc_series.insert(0, "CLASS", row["xray_class"])
        bc_data = pd.concat([bc_data, bc_series])

    # Get M and X class flares, round down their minutes.
    m_df = mx_info_df.loc[mx_info_df["xray_class"] == "M"]
    print("Class M Flares Shape: ", m_df.shape)
    x_df = mx_info_df.loc[mx_info_df["xray_class"] == "X"]
    print("Class X Flares Shape: ", x_df.shape)
    mx_info = pd.concat([m_df, x_df])
    mx_info["time_start"] = \
        mx_info["time_start"].apply(floor_minute)

    # Find the respective timestamp in the MX data file.
    mx_data = pd.DataFrame()
    for index, row in mx_info.iterrows():
        timestamp = row["time_start"]
        mx_series = mx_properties_df.loc[
            abc_properties_df["T_REC"] == timestamp].head(1)
        # mx_series.insert(0, "CLASS", row["xray_class"])
        mx_data = pd.concat([mx_data, mx_series])

    # Find timepoints where multiple flares occur in complete set,
    # regardless of flare class.
    # Note: Complete set is sorted by flare class, starting from B, C, M, and X.
    # 1. Take the end time of the event to be the end time of the actual flare.
    # 2. Take the start time of the event to be 24 hours prior to the end time.
    # 3. For all other flares in the complete set:
    #    i. Get the other flare's range like above, then determine if both
    #    flares' ranges have any overlap -- if so, then append it to a list.
    flare_info = pd.concat([bc_info, mx_info], ignore_index=True)
    flare_info.drop("Unnamed: 0", axis=1, inplace=True)

    b_df = flare_info.loc[flare_info["xray_class"] == "B"]
    c_df = flare_info.loc[flare_info["xray_class"] == "C"]
    m_df = flare_info.loc[flare_info["xray_class"] == "M"]
    x_df = flare_info.loc[flare_info["xray_class"] == "X"]


    def info_to_data(info, data):
        df = pd.DataFrame()
        for index, row in info.iterrows():
            timestamp = row["time_start"]
            df_sort = data.iloc[
                (data['T_REC'] - timestamp).abs().argsort()[:1]]
            df = pd.concat([df, df_sort])
        return df


    info_to_data(b_df, bc_data)
    info_to_data(b_df, bc_data)
    info_to_data(b_df, bc_data)
    b_data_df = info_to_data(b_df, bc_data)
    c_data_df = info_to_data(c_df, bc_data)
    m_data_df = info_to_data(m_df, mx_data)
    x_data_df = info_to_data(x_df, mx_data)


    # Plot PCA
    flare_dataframes = [b_data_df, c_data_df, m_data_df, x_data_df]
    flare_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(2, 2)
    pc_labels = [f"PC{i}" for i in range(1, 17 + 1)]
    i, j = 0, 0
    for df, label, indices in zip(flare_dataframes, CLASS_LABELS, flare_indices):
        i, j = indices
        df.drop(["T_REC", "NOAA_AR"], axis=1, inplace=True)
        pca = PCA()
        pca.fit_transform(MinMaxScaler().fit_transform(df))
        ev = pca.explained_variance_ratio_
        ax[i, j].set_title(f"Class {label} PCA ({df.shape[0]} Flares)")
        ax[i, j].set_xlabel("Principal Components")
        ax[i, j].set_ylabel("Explained Variance Ratio")
        ax[i, j].set_xticks(range(17), pc_labels, fontsize=8, rotation="vertical")
        ax[i, j].bar(range(len(ev)), list(ev * 100),
                  align="center", color="b")
    fig.tight_layout()
    fig.show()


    print(b_data_df)
    print(c_data_df)
    print(m_data_df)
    print(x_data_df)




    # MULTIPLE FLARES BELOW

    # multiple_flares = []
    # for index in range(flare_info.shape[0]):
    #     flare = {"flare_index": index, "overlapping_flares": []}
    #     multiple_flares.append(flare)
    # flare_matrix = np.zeros(shape=(flare_info.shape[0], flare_info.shape[0]),
    #                         dtype=int)
    #
    # print(flare_info.shape)

    # for index1, row1 in flare_info.iterrows():
    #     flare = multiple_flares[int(index1)]
    #     print(flare)
    #     flare_class1 = row1["xray_class"]
    #     print(index1)
    #     time_end1 = row1["time_end"]
    #     time_start1 = time_end1 - datetime.timedelta(1)
    #     for index2, row2 in flare_info.iterrows():
    #         flare_class2 = row2["xray_class"]
    #         # Don't count the same flare.
    #         if index1 == index2:
    #             continue
    #         # Only look for flares in the same class.
    #         if flare_class1 != flare_class2:
    #             continue
    #         time_end2 = row2["time_end"]
    #         time_start2 = time_end2 - datetime.timedelta(1)
    #         flares_overlap = (time_start1 <= time_start2 <= time_end1) or (
    #                 time_start1 <= time_end2 <= time_end1)
    #         if flares_overlap:
    #             flare_matrix[index1][index2] = 1
    #             flare["overlapping_flares"].append(f"{index2},{flare_class2}")

    # with open("multiple_solar_events/overlapping_flares_same.csv", "w") as f:
    #     writer = csv.DictWriter(f, fieldnames=["flare_index",
    #                                            "overlapping_flares"])
    #     writer.writeheader()
    #     writer.writerows(multiple_flares)

    # im = plt.imshow(flare_matrix, cmap="binary")
    #
    # plt.colorbar(im)
    # plt.title("Overlapping Flare Events (Same Classes)")
    # plt.show()
    # print("Multiple Flares", multiple_flares)
    # multiple_flares_df = pd.DataFrame()
    # for key, value in multiple_flares.items():
    #     series = pd.Series(value, name=key)
    #     multiple_flares_df = pd.concat([multiple_flares_df, series])
    # print(multiple_flares_df)
    # multiple_flares_df.to_csv("multiple_flares.csv")

    # print("Time Start:", time_start)
    # print("Time End:", time_end)


if __name__ == "__main__":
    main()
