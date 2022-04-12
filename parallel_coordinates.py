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
    time_range = 12  # Valid: 1 to 48 hours

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

    # Get first 40 of B and C class flares, round down their minutes.
    b_df = abc_info_df.loc[abc_info_df["xray_class"] == "B"].head(58)
    c_df = abc_info_df.loc[abc_info_df["xray_class"] == "C"].head(56)
    bc_info = pd.concat([b_df, c_df])
    bc_info["time_start"] = \
        bc_info["time_start"].apply(floor_minute)

    # Find the respective timestamp in the ABC data file.
    bc_data = pd.DataFrame()
    for index, row in bc_info.iterrows():
        timestamp = row["time_start"]
        bc_series = abc_properties_df.loc[
            abc_properties_df["T_REC"] == timestamp].head(1)
        bc_series.insert(0, "CLASS", row["xray_class"])
        bc_data = pd.concat([bc_data, bc_series])

    # Get first 40 of M and X class flares, round down their minutes.
    m_df = mx_info_df.loc[mx_info_df["xray_class"] == "M"].head(45)
    x_df = mx_info_df.loc[mx_info_df["xray_class"] == "X"].head(100)
    mx_info = pd.concat([m_df, x_df])
    mx_info["time_start"] = \
        mx_info["time_start"].apply(floor_minute)

    # Find the respective timestamp in the MX data file.
    mx_data = pd.DataFrame()
    for index, row in mx_info.iterrows():
        timestamp = row["time_start"]
        mx_series = mx_properties_df.loc[
            mx_properties_df["T_REC"] == timestamp].head(1)
        mx_series.insert(0, "CLASS", row["xray_class"])
        mx_data = pd.concat([mx_data, mx_series])

    # Combine BC and MX class flares into one dataframe and plot.
    colors = ["cyan", "lime", "orange", "red"]
    columns = FLARE_PROPERTIES
    flares_df = pd.concat([bc_data, mx_data])
    flares_df[columns] = MinMaxScaler().fit_transform(flares_df[columns])
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(25, 12))
    pd.plotting.parallel_coordinates(flares_df.drop("T_REC", axis=1).
                                     drop("NOAA_AR", axis=1),
                                     "CLASS", color=colors)
    fig.tight_layout()
    fig.show()
    fig.savefig("parallel_coordinates")













    # COMMENT OUT BELOW CODE FOR SINGLE PLOTS

    # # Find corresponding ending index in properties dataframe.
    # abc_end_series = abc_properties_df.loc[
    #     abc_properties_df["T_REC"] == abc_timestamp]
    # abc_end_index = abc_end_series.loc[
    #     abc_end_series['NOAA_AR'] == abc_noaa].index.tolist()[0]
    # mx_end_series = mx_properties_df.loc[
    #     mx_properties_df["T_REC"] == mx_timestamp]
    # mx_end_index = mx_end_series.loc[
    #     mx_end_series['NOAA_AR'] == mx_noaa].index.tolist()[0]
    #
    # # Find corresponding starting index in properties dataframe, if it exists.
    # abc_start_index = abc_end_index
    # for i in range(time_range * 5 - 1):
    #     if abc_end_index - i >= 0:
    #         if abc_properties_df["NOAA_AR"][abc_end_index - i] == abc_noaa:
    #             abc_start_index = abc_end_index - i
    # mx_start_index = mx_end_index
    # for i in range(time_range * 5 - 1):
    #     if mx_end_index - i >= 0:
    #         if mx_properties_df["NOAA_AR"][mx_end_index - i] == abc_noaa:
    #             mx_start_index = mx_end_index - i
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