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
import datetime as dt
from sklearn.linear_model import LinearRegression
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
    abc_properties_df.drop(to_drop, inplace=True, axis=1)
    mx_properties_df = pd.read_csv("Data_MX.csv")
    mx_properties_df.drop(to_drop, inplace=True, axis=1)

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

    b_df = info_df.loc[info_df["xray_class"] == "B"]
    c_df = info_df.loc[info_df["xray_class"] == "C"]
    m_df = info_df.loc[info_df["xray_class"] == "M"]
    x_df = info_df.loc[info_df["xray_class"] == "X"]

    single_dfs = [(x_df, "X"), (m_df, "M"), (b_df, "B"), (c_df, "C")]

    bc_info = pd.concat([
        b_df,
        c_df
    ])
    mx_info = pd.concat([
        m_df,
        x_df
    ])
    bx_info = pd.concat([
        b_df,
        x_df
    ])
    pair_dfs = [(mx_info, "MX"), (bx_info, "BX"), (bc_info, "BC")]

    dfs = single_dfs + pair_dfs + [(pd.concat([bc_info, mx_info]), "BCMX")]

    info_df.reset_index(inplace=True)
    for info_df, label in dfs:
        series_df = pd.DataFrame()
        info_df.reset_index(inplace=True)
        print(info_df)

        for index, row in info_df.iterrows():
            print(index, "/", info_df.shape[0])
            end_timestamp = row["time_start"]
            if end_timestamp < pd.Timestamp(2010, 5, 1):
                continue
            flare_class = row["xray_class"]
            if flare_class in ["B", "C"]:
                properties = abc_properties_df
            else:
                properties = mx_properties_df

            start_timestamp = end_timestamp - datetime.timedelta(0, 3600 * 6)
            df_start = properties.iloc[
                (properties['T_REC'] - start_timestamp).abs().argsort()[:1]]
            df_end = properties.iloc[
                (properties['T_REC'] - end_timestamp).abs().argsort()[:1]]
            start_index = df_start.iloc[0].name
            end_index = df_end.iloc[0].name
            df = properties_df[start_index:end_index].drop(
                ["T_REC", "NOAA_AR"], axis=1)
            # dataframes.append((range_df, row["xray_class"]))
            mean_df = df.mean().to_frame().T
            if mean_df.isnull().values.any():
                continue
            print(mean_df)

            mean_df["xray_class"] = [flare_class]

            series_df = pd.concat([
                series_df, mean_df], ignore_index=True)

            print(series_df)

        series_df.dropna(inplace=True)
        series_df.reset_index(inplace=True)
        series_df.drop("index", axis=1, inplace=True)
        print(series_df)

        # n = 7
        # pc_labels = [f"PC{i}" for i in range(1, n + 1)]
        # pca = PCA(n_components=n)
        # flare_pca = pca.fit_transform(MinMaxScaler().fit_transform(series_df))
        # pca_df = pd.DataFrame(data=flare_pca, columns=pc_labels)
        # print(pca_df, pca_df.columns)
        # pca_df["xray_class"] = pd.Series(data_df["xray_class"])

        if len(label) <= 1:
            color = "ABSNJZH"
        else:
            color = "xray_class"
        fig = px.scatter_3d(series_df, x="USFLUX", y="R_VALUE", z="MEANGAM", color=color,
                            title=f"Non-Correlative Parameters for {label} Class Flares ({series_df.shape[0]} Flares)")
        fig.write_html(f"correlation/6h/{label}_mean_3d.html")


        # def info_to_data(info, data):
        #     series_df = pd.DataFrame()
        #     df = pd.DataFrame()
        #     print(info)
        #     for index, row in info.iterrows():
        #         noaa_ar = info_df["nar"][index]
        #         print(f"{index}/{info.shape[0]}")
        #         end_timestamp = row["time_start"]
        #         start_timestamp = end_timestamp - datetime.timedelta(1)
        #         df = data[(data['T_REC'] < start_timestamp) & (data['T_REC'] > end_timestamp)]
        #         print(df)
        #         exit(1)
        #         series = data["T_REC"].between_time(start_timestamp, end_timestamp)
        #         print(series)
            # df.reset_index(inplace=True)
            # df.drop("index", axis=1, inplace=True)
            # return series_df


        # df = info_to_data(info_df, properties_df)
        # df2 = df.copy(deep=True)
        # df = df.drop(["xray_class", "T_REC", "NOAA_AR"], axis=1)
        # cm = df.corr().abs()
        # upper_tri = cm.where(
        #     np.triu(np.ones(cm.shape), k=1).astype(np.bool))
        # to_drop = [column for column in upper_tri.columns if
        #            any(upper_tri[column] >= 0.75)]
        # print(to_drop)
        # df1 = df2.drop(to_drop, axis=1)
        # df1.to_csv(f"correlation/6h/{label}_reduced.csv")
        # df1 = df1.drop(["xray_class", "T_REC", "NOAA_AR"], axis=1)
        # cm = df1.corr()
        # print(cm)
        #
        # properties = list(set(FLARE_PROPERTIES) - set(to_drop))
        #
        # plt.rcParams["figure.figsize"] = (19, 11)
        # sns.heatmap(cm, annot=True, cmap="RdYlBu", cbar=True, fmt=".2f",
        #             square=True, xticklabels=properties, yticklabels=properties)
        # plt.title(f"{label} Flare Correlation Matrix (Time Start) - Exhaustive")
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    main()
