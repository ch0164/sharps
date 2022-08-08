import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib
from astropy.io import fits
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import sunpy.map
import sunpy.io
from IPython.display import Image
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

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

    # Define input for flare.
    flare_index = 9  # Valid: 0 to 765
    time_range = 48  # Valid: 1 to 48 hours

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
        nar = row["nar"]
        bc_series = abc_properties_df.loc[
            abc_properties_df["T_REC"] == timestamp].head(1)
        if bc_series.empty:
            continue
        elif bc_series.loc[bc_series["NOAA_AR"] == nar].empty:
            continue
        bc_end_index = bc_series.loc[bc_series["NOAA_AR"] == nar].index.tolist()[0]
        bc_start_index = bc_end_index

        # Get defined range before flare.
        for i in range(time_range * 5 - 1):
            if bc_end_index - i >= 0:
                if abc_properties_df["NOAA_AR"][bc_end_index - i] == nar:
                    bc_start_index = bc_end_index - i
        bc_range = abc_properties_df.iloc[bc_start_index:bc_end_index]

        # Get the average for this flare.
        bc_range_mean = bc_range.drop("T_REC", axis=1).drop("NOAA_AR", axis=1)\
            .mean().to_frame().T
        bc_range_mean.insert(0, "CLASS", row["xray_class"])
        bc_data = pd.concat([bc_data, bc_range_mean])
    print(bc_data)
    print(bc_data[bc_data["CLASS"] == "B"])

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
        nar = row["nar"]
        mx_series = mx_properties_df.loc[
            mx_properties_df["T_REC"] == timestamp].head(1)
        if mx_series.empty:
            continue
        elif mx_series.loc[mx_series["NOAA_AR"] == nar].empty:
            continue
        mx_end_index = mx_series.loc[mx_series["NOAA_AR"] == nar].index.tolist()[0]
        mx_start_index = mx_end_index

        # Get defined range before flare.
        for i in range(time_range * 5 - 1):
            if mx_end_index - i >= 0:
                if mx_properties_df["NOAA_AR"][mx_end_index - i] == nar:
                    mx_start_index = mx_end_index - i
        mx_range = mx_properties_df.iloc[mx_start_index:mx_end_index]

        # Get the average for this flare.
        mx_range_mean = mx_range.drop("T_REC", axis=1).drop("NOAA_AR", axis=1)\
            .mean().to_frame().T
        mx_range_mean.insert(0, "CLASS", row["xray_class"])
        mx_data = pd.concat([mx_data, mx_range_mean])
    # print(mx_data)
    print(mx_data[mx_data["CLASS"] == "X"])

        # mx_data = pd.concat([mx_data, mx_series])

    # Combine BC and MX class flares into one dataframe and plot.
    # Shapes:
    # Class B Flares Shape:  (423, 6)
    # Class C Flares Shape:  (577, 6)
    # Class M Flares Shape:  (715, 6)
    # Class X Flares Shape:  (51, 6)
    colors = ["cyan", "lime", "orange", "red"]
    columns = FLARE_PROPERTIES
    time_label = f"Flare {time_range}h Range"
    flares_df = pd.concat([bc_data, mx_data])

    b_df = flares_df.loc[flares_df["CLASS"] == "B"]
    c_df = flares_df.loc[flares_df["CLASS"] == "C"]
    m_df = flares_df.loc[flares_df["CLASS"] == "M"]
    x_df = flares_df.loc[flares_df["CLASS"] == "X"]

    print(b_df, c_df, m_df, x_df)

    print(b_df.mean())

    average_df = b_df.mean()

    # Plot specified flare properties over the specified time.
    for flare_class, flare_df in zip(CLASS_LABELS, [b_df, c_df, m_df, x_df]):
        fig, ax = plt.subplots(6, 3, figsize=(18, 20))
        row, col = 0, 0
        if flare_class in ["B", "C"]:
            properties_df = abc_properties_df
        else:
            properties_df = mx_properties_df

        for flare_property in FLARE_PROPERTIES:
            property_df = properties_df[["T_REC", flare_property]]
            print(property_df["T_REC"])
            property_df.iloc[abc_start_index:abc_end_index].plot_pca(
                x="T_REC", y=flare_property, ax=ax[row, col], legend=False)
            ax[row, col].set_ylabel(flare_property)
            ax[row, col].set_title(
                f"Total {flare_property} from {properties_df['T_REC'].values[0]}")

            col += 1
            if col == 3:
                col = 0
                row += 1

        fig.tight_layout()
        fig.show()
        fig.savefig(f"{flare_class}_average_plot")



    # scaler = MinMaxScaler()
    # flares_df[columns] = scaler.fit_transform(flares_df[columns])
    # print(scaler.scale_)
    #
    # # Plot the complete set of flares for all classes.
    # plt.style.use('dark_background')
    # fig = plt.figure(figsize=(25, 12))
    # pd.plotting.parallel_coordinates(flares_df,
    #                                  "CLASS", color=colors)
    # fig.tight_layout()
    # fig.suptitle(f"All Flares Complete ({time_label})", fontsize=20)
    # fig.show()
    # fig.savefig(f"parallel_coordinates/{time_label}/complete_all_classes.png")
    #
    # # For each flare, plot their complete set individually.
    # # Then, plot a random subset of them individually,
    # # as well as all together.
    # mean_df = pd.DataFrame()
    # subset_df = pd.DataFrame()
    # n = 28
    # for flare_class, color in zip(CLASS_LABELS, colors):
    #     fig = plt.figure(figsize=(25, 12))
    #     class_df = flares_df[flares_df["CLASS"] == flare_class]
    #     pd.plotting.parallel_coordinates(class_df,
    #                                      "CLASS", color=[color])
    #     fig.tight_layout()
    #     fig.suptitle(f"{flare_class} Complete ({time_label})",
    #                  fontsize=20)
    #     fig.show()
    #     fig.savefig(f"parallel_coordinates/{time_label}/complete_{flare_class}_class.png")
    #
    #
    #     fig = plt.figure(figsize=(25, 12))
    #     # X Class flares are limited.
    #     if "X" in flare_class and n > 28:
    #         class_subset_df = class_df.sample(n=28)
    #     else:
    #         class_subset_df = class_df.sample(n=n)
    #     pd.plotting.parallel_coordinates(class_subset_df,
    #                                      "CLASS", color=[color])
    #     fig.tight_layout()
    #     fig.suptitle(f"{flare_class} Random Sample, n = {n} ({time_label})", fontsize=20)
    #     fig.show()
    #     fig.savefig(f"parallel_coordinates/{time_label}/subset{n}_{flare_class}_class.png")
    #
    #     subset_df = pd.concat([subset_df, class_subset_df])
    #
    #     # Plot the average for the complete set of flares for all classes.
    #     fig = plt.figure(figsize=(25, 12))
    #     class_mean_df = class_df.mean().to_frame().T
    #     class_mean_df["CLASS"] = flare_class
    #     mean_df = pd.concat([mean_df, class_mean_df])
    #
    #
    #
    # fig = plt.figure(figsize=(25, 12))
    # pd.plotting.parallel_coordinates(subset_df,
    #                                  "CLASS", color=colors)
    # fig.tight_layout()
    # fig.suptitle(f"All Flares Random Sample, n = {n} ({time_label})", fontsize=20)
    # fig.show()
    # fig.savefig(f"parallel_coordinates/{time_label}/subset{n}_all_classes.png")
    #
    # fig = plt.figure(figsize=(25, 12))
    # pd.plotting.parallel_coordinates(mean_df,
    #                                  "CLASS", color=colors)
    # fig.tight_layout()
    # fig.suptitle(f"Mean ({time_label})", fontsize=20)
    # fig.show()
    # fig.savefig(f"parallel_coordinates/{time_label}/complete_mean_all_classes.png")
    #
    #
    # # Plot boxplots for all classes.
    # plt.style.use('default')
    # for flare_class in CLASS_LABELS:
    #     fig = plt.figure(figsize=(25, 12))
    #     if flare_class in ["B", "C"]:
    #         flare_data = bc_data[bc_data["CLASS"] == flare_class]
    #     else:
    #         flare_data = mx_data[mx_data["CLASS"] == flare_class]
    #     flare_data[columns] = MinMaxScaler().fit_transform(flare_data[columns])
    #     flare_data.boxplot(columns)
    #     fig.suptitle(f"{flare_class} Class Flare ({time_label})\n"
    #                  f"{flare_data.shape[0]} datapoints", fontsize=20)
    #     fig.tight_layout()
    #     fig.show()
    #     fig.savefig(f"parallel_coordinates/{time_label}/{flare_class}_boxplot.png")






    # COMMENT OUT BELOW CODE FOR SINGLE PLOTS

    #
    # # Plot specified flare properties over the specified time.
    # fig, ax = plt.subplots(6, 3, figsize=(18, 20))
    # row, col = 0, 0
    # pd.plotting.parallel_coordinates(abc_properties_df.iloc[abc_start_index], "T_REC")
    # plt.tight_layout()
    # plt.show()
    # for flare_property in FLARE_PROPERTIES:
    #     property_df = properties_df[["T_REC", flare_property]]
    #     print(property_df["T_REC"])
    #     property_df.iloc[abc_start_index:abc_end_index].plot(
    #         x="T_REC", y=flare_property, ax=ax[row, col], legend=False)
    #     ax[row, col].set_ylabel(flare_property)
    #     ax[row, col].set_title(
    #         f"Total {flare_property} from {properties_df['T_REC'].values[0]}")
    #
    #     col += 1
    #     if col == 3:
    #         col = 0
    #         row += 1
    #
    # fig.tight_layout()
    # fig.show()
    # fig.savefig("abc_output")


    # print(abc_properties_df)
    # print(mx_properties_df)

    # print(abc_info_df)
    # print(mx_info_df)














if __name__ == "__main__":
    main()