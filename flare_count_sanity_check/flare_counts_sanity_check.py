import datetime
from datetime import datetime as dt_obj

import pandas
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
import os

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

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
FLARE_PROPERTIES += ['d_l_f', 'g_s', 'slf']

FLARE_LABELS = ["B", "C", "M", "X"]
FLARE_COLORS = ["blue", "green", "orange", "red"]
COINCIDENCES = ["all", "coincident", "noncoincident"]


# COINCIDENCES = ["all"]


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


def flare_to_num(flare_label):
    if flare_label in "ABC":
        return 0
    elif flare_label in "MX":
        return 1


def time_range_list(time_interval, start_time):
    time_step_range = range(time_interval * 5 + 1)
    times = [(start_time + datetime.timedelta(0, 12 * 60 * i)) for i in time_step_range]
    return times, time_step_range


time_interval = 12
abc_properties_df = pd.read_csv("../Data_ABC_with_Korsos_parms.csv")
mx_properties_df = pd.read_csv("../Data_MX_with_Korsos_parms.csv")

flare_counts = {label: 0 for label in FLARE_LABELS}


def main():
    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    new_info_df = pd.read_csv(f"../all_flare_info_2.csv")
    old_info_df = pd.concat([pd.read_csv("../ABC_list.txt"),
                             pd.read_csv("../MX_list.txt")])
    old_info_df["xray_class"] = \
        old_info_df["xray_class"].apply(classify_flare)
    for time_string in ["time_start", "time_peak", "time_end"]:
        new_info_df[time_string] = \
            new_info_df[time_string].apply(parse_tai_string)
        old_info_df[time_string] = \
            old_info_df[time_string].apply(parse_tai_string)

    for info_df, label in zip([new_info_df, old_info_df], ["all_flare_catalog", "curated_flare_catalog"]):
        missing = {
            "B": {i: 0 for i in range(0, 60 + 1)},
            "C": {i: 0 for i in range(0, 60 + 1)},
            "M": {i: 0 for i in range(0, 60 + 1)},
            "X": {i: 0 for i in range(0, 60 + 1)}
        }

        cols = info_df.columns
        all_good, missing_time = pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
        for flare_index, row in info_df.iterrows():
            print(flare_index, "/", info_df.shape[0])
            # Find NOAA AR number and timestamp from user input in info dataframe.
            noaa_ar = row["nar"]
            timestamp = floor_minute(row["time_start"]) - datetime.timedelta(hours=12)
            times, time_range = time_range_list(time_interval, timestamp)
            flare_class = row["xray_class"]

            if flare_class in ["B", "C"]:
                properties_df = abc_properties_df
            else:
                properties_df = mx_properties_df

            # Find corresponding ending index in properties dataframe.
            fail = False
            for t_rec in times:
                x = properties_df.loc[properties_df['T_REC'] == t_rec]
                if x.empty:
                    missing[flare_class][t_rec.hour] += 1
                    missing_time.loc[len(missing_time)] = row
                    fail = True
                    break

            if not fail:
                end_series = properties_df.loc[
                    properties_df["T_REC"] == timestamp]
                if not end_series.loc[(end_series['NOAA_AR'] == noaa_ar) & (end_series["QUALITY"] == 0)].empty:
                    all_good.loc[len(all_good)] = row

        for df, df_label in zip([all_good, missing_time], ["good", "missing_time"]):
            b = df.loc[df["xray_class"] == "B"].shape[0]
            c = df.loc[df["xray_class"] == "C"].shape[0]
            m = df.loc[df["xray_class"] == "M"].shape[0]
            x = df.loc[df["xray_class"] == "X"].shape[0]

            with open(f"{label}_{df_label}_flares.txt", "w", newline="\n") as f:
                f.write("Flare Counts\n")
                f.write('-' * 50 + "\n")
                f.write(f"B: {b}\n")
                f.write(f"C: {c}\n")
                f.write(f"M: {m}\n")
                f.write(f"X: {x}\n")

            for flare_class in FLARE_LABELS:
                hour = 10
                with open(f"{label}_{df_label}_{flare_class.lower()}_flare_10_22h_histogram.txt", "w",
                          newline="\n") as f:
                    f.write(f"{flare_class} 10h-22h Flare Histogram\n")
                    f.write('-' * 50 + "\n")
                    for i in range(0, 61, 10):
                        sum = sum(missing[flare_class][i:i+10])
                        f.write(f"{hour}h-{hour + 1}: {sum}\n")
                        hour += 2


if __name__ == "__main__":
    main()
