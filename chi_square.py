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
    labels = ["B", "C", "M", "X"]
    n = len(UNIQUE_PROPERTIES)

    for is_coincident in ["Coincident", "Noncoincident"]:
        # plt.clf()
        # for label in labels:
        #     df = pd.read_csv(f"24_average_{label.lower()}_{is_coincident.lower()}.csv")
        #     new_df = pd.DataFrame(columns=UNIQUE_PROPERTIES, index=range(24))
        #     bins = []
        #     for hour in range(0, 24):
        #         if hour == 0:
        #             bin_size = 4
        #             increment = 1
        #         else:
        #             bin_size = 5
        #             increment = 0
        #         bin_ = [5 * hour + i + increment for i in range(0, bin_size)]
        #         bins.append(bin_)
        #
        #     for hour in range(0, 24):
        #         bin_ = bins[hour]
        #         data_points = df.iloc[bin_, :]
        #         for flare_property in UNIQUE_PROPERTIES:
        #             new_df[flare_property][hour] = data_points[
        #                 flare_property].mean()
        #     print(new_df)
        #     new_df.to_csv(f"chi_square/{is_coincident.lower()}/24_average_{label.lower()}_{is_coincident.lower()}_binned.csv")

        for label in labels:
            df = pd.read_csv(f"chi_square/{is_coincident.lower()}/24_average_{label.lower()}_{is_coincident.lower()}_binned.csv")
            x2 = np.zeros((n, n), dtype=float)

            def property_to_num(property):
                return UNIQUE_PROPERTIES.index(property)

            for property1 in UNIQUE_PROPERTIES:
                other_properties = sorted(list(set(UNIQUE_PROPERTIES) - {property1}))
                print(property1)
                print(other_properties)
                for property2 in other_properties:
                    f_obs = df[property1]
                    f_exp = df[property2]
                    f_diff = ((f_obs - f_exp)**2 / f_exp).sum()
                    x2[property_to_num(property1)][property_to_num(property2)] = f_diff


            # print(x_df)
            simple_matrix = np.zeros(x2.shape)
            for i, row in enumerate(x2):
                for j, value in enumerate(row):
                    if value >= 33.196:
                        simple_matrix[i][j] = 1
                    if value >= 36.415:
                        simple_matrix[i][j] = 2
                    if value >= 42.980:
                        simple_matrix[i][j] = 3


            # plt.figure(figsize=(15, 15))
            # cmap = ListedColormap(["red", "green", "blue"])
            # ax = sns.heatmap(simple_matrix, annot=False, cmap=cmap, cbar=False, #fmt=".3f",
            #             square=True, xticklabels=UNIQUE_PROPERTIES, yticklabels=UNIQUE_PROPERTIES)
            # colorbar = ax.collections[0].colorbar
            # colorbar.set_ticks([0, 1, 2])
            # colorbar.set_ticklabels(['Accept', 'Reject (95% Confidence)', 'Reject (95% Confidence)'])

            colors = {"gray": 0, "red": 1, "yellow": 2, "blue": 3}
            l_colors = sorted(colors, key=colors.get)
            import matplotlib.colors as c
            cMap = c.ListedColormap(l_colors)
            ax = sns.heatmap(simple_matrix, cmap=l_colors, vmin=0, vmax=len(colors), linecolor="black", linewidth=2,
                             annot=x2, fmt=".3f",
                        square=True, xticklabels=UNIQUE_PROPERTIES, yticklabels=UNIQUE_PROPERTIES)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([0, 1, 2, 3])
            colorbar.set_ticklabels(['Accept', 'Reject (90% Confidence)',
                                     'Reject (95% Confidence)',
                                     'Reject (99% Confidence)'])


            plt.title(f"Chi-Square Test of Parameters\n over 24h Average Timeline ({label} Class {is_coincident} Flares)")
            # plt.title(f"{flare_class} Flares (Mean {time_range}h Time Series)")
            plt.tight_layout()
            plt.savefig(f"chi_square/{is_coincident.lower()}/noncorrelative_{label.lower()}_24h_average_chi_square")
            plt.show()
            print(x2)


        # for property1 in UNIQUE_PROPERTIES:
        #     for property2 in list(set(FLARE_PROPERTIES) - {property1}):
        #         property1_df = data_points[property1]
        #         property2_df = data_points[property2]
        #         print(property1_df)
        #         print(property2_df)
        # print(data_points)




    # abc_properties_df = pd.read_csv("Data_ABC.csv")
    # abc_properties_df.dropna(inplace=True)
    # abc_properties_df.drop(to_drop, inplace=True, axis=1)
    # mx_properties_df = pd.read_csv("Data_MX.csv")
    # mx_properties_df.dropna(inplace=True)
    # mx_properties_df.drop(to_drop, inplace=True, axis=1)
    #
    # info_df = pd.read_csv("all_flares.txt")
    # info_df.drop(["hec_id", "lat_hg", "long_hg", "long_carr", "optical_class"], axis=1, inplace=True)
    #
    # # Convert time strings to datetime objects for cleaned info data.
    # for time_string in ["time_start", "time_peak", "time_end"]:
    #     info_df[time_string] = \
    #         info_df[time_string].apply(parse_tai_string)
    #
    # # Convert T_REC string to datetime objects.
    # abc_properties_df["T_REC"] = \
    #     abc_properties_df["T_REC"].apply(parse_tai_string)
    # mx_properties_df["T_REC"] = \
    #     mx_properties_df["T_REC"].apply(parse_tai_string)
    # # properties_df = pd.concat([abc_properties_df, mx_properties_df])
    #
    # # Label flares by B, C, M, and X.
    # info_df["xray_class"] = \
    #     info_df["xray_class"].apply(classify_flare)
    #
    # b_df = info_df.loc[info_df["xray_class"] == "B"]
    # c_df = info_df.loc[info_df["xray_class"] == "C"]
    # m_df = info_df.loc[info_df["xray_class"] == "M"]
    # x_df = info_df.loc[info_df["xray_class"] == "X"]
    #
    # single_dfs = [(x_df, "X"), (m_df, "M"), (b_df, "B"), (c_df, "C")]
    #
    # bc_info = pd.concat([
    #     b_df,
    #     c_df
    # ])
    # mx_info = pd.concat([
    #     m_df,
    #     x_df
    # ])
    # bx_info = pd.concat([
    #     b_df,
    #     x_df
    # ])
    # pair_dfs = [(mx_info, "MX"), (bx_info, "BX"), (bc_info, "BC")]
    #
    # dfs = single_dfs + pair_dfs + [(pd.concat([bc_info, mx_info]), "BCMX")]
    #
    # info_df.reset_index(inplace=True)
    #
    # bin_sep = 60
    # seconds = [bin_sep * i * 60 for i in range(1, 25)]
    # bins = [(start, start + 1) for start in range(0, 24)]
    # seconds_diffs = [(start * 3600, end * 3600) for start, end in bins]
    #
    # print(x_df.to_string())
    #
    # mean_df = mx_properties_df.mean().to_frame().T
    # print(mean_df)
    #
    #
    #
    # for index, row in x_df.iterrows():
    #     for start, end in seconds_diffs:
    #         new_df = mx_properties_df.loc[mx_properties_df["NOAA_AR"] == row["nar"]]
    #
    #         end_timestamp = row["time_start"] - datetime.timedelta(0, start)
    #         start_timestamp = row["time_start"] - datetime.timedelta(0, end)
    #         df_start = mx_properties_df.iloc[
    #             (mx_properties_df['T_REC'] - start_timestamp).abs().argsort()[
    #             :1]]
    #         df_end = mx_properties_df.iloc[
    #             (mx_properties_df['T_REC'] - end_timestamp).abs().argsort()[
    #             :1]]
    #         start_index = df_start.iloc[0].name
    #         end_index = df_end.iloc[0].name
    #
    #         df = mx_properties_df[start_index:end_index] \
    #             # .drop(
    #             # ["T_REC", "NOAA_AR"], axis=1)
    #         print(df)
    #         mean_df = df.mean().to_frame().T
    #         print(mean_df)
    #         exit(1)













    # for info_df, label in dfs:
    #     series_df = pd.DataFrame()
    #     info_df.reset_index(inplace=True)
    #     print(info_df)
    #
    #     timepoint_df = pd.DataFrame()
    #     for index, row in info_df.iterrows():
    #         if row["xray_class"] in ["B", "C"]:
    #             properties_df = abc_properties_df
    #         else:
    #             properties_df = mx_properties_df
    #         properties = properties_df.loc[properties_df["NOAA_AR"] == row["nar"]]
    #         for start, end in seconds_diffs:
    #             end_timestamp = row["time_start"] - datetime.timedelta(0, start)
    #             start_timestamp = row["time_start"] - datetime.timedelta(0, end)
    #             df_start = properties.iloc[
    #                 (properties['T_REC'] - start_timestamp).abs().argsort()[
    #                 :1]]
    #             df_end = properties.iloc[
    #                 (properties['T_REC'] - end_timestamp).abs().argsort()[
    #                 :1]]
    #             start_index = df_start.iloc[0].name
    #             end_index = df_end.iloc[0].name
    #
    #             df = properties_df[start_index:end_index]\
    #                 .drop(
    #                 ["T_REC", "NOAA_AR"], axis=1)
    #             print(df)
    #             mean_df = df.mean().to_frame().T
    #             if mean_df.isnull().values.any():
    #                 continue
    #             print(mean_df)
    #
    #             # mean_df["xray_class"] = row["xray_class"]
    #
    #             print("Mean DF", mean_df)
    #
    #             timepoint_df = pd.concat([
    #                 timepoint_df, mean_df], ignore_index=True)
    #
    #             print("Timepoint DF", timepoint_df)
    #
    #
    #             if index == 5:
    #                 exit(1)
    #         print(index, "/", info_df.shape[0])
    #         end_timestamp = row["time_start"]
    #         if end_timestamp < pd.Timestamp(2010, 5, 1):
    #             continue
    #         flare_class = row["xray_class"]
    #         if flare_class in ["B", "C"]:
    #             properties = abc_properties_df
    #         else:
    #             properties = mx_properties_df
    #
    #         start_timestamp = end_timestamp - datetime.timedelta(0, 3600 * 6)
    #         df_start = properties.iloc[
    #             (properties['T_REC'] - start_timestamp).abs().argsort()[:1]]
    #         df_end = properties.iloc[
    #             (properties['T_REC'] - end_timestamp).abs().argsort()[:1]]
    #         start_index = df_start.iloc[0].name
    #         end_index = df_end.iloc[0].name
    #         df = properties_df[start_index:end_index].drop(
    #             ["T_REC", "NOAA_AR"], axis=1)
    #         # dataframes.append((range_df, row["xray_class"]))
    #         mean_df = df.mean().to_frame().T
    #         if mean_df.isnull().values.any():
    #             continue
    #         print(mean_df)
    #
    #         mean_df["xray_class"] = [flare_class]
    #
    #         series_df = pd.concat([
    #             series_df, mean_df], ignore_index=True)
    #
    #         print(series_df)
    #
    #     series_df.dropna(inplace=True)
    #     series_df.reset_index(inplace=True)
    #     series_df.drop("index", axis=1, inplace=True)
    #     print(series_df)



if __name__ == "__main__":
    main()