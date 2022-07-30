import datetime
from datetime import datetime as dt_obj

import numpy as np
import pandas as pd
import os

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

FLARE_LABELS = ["B", "C", "M", "X"]
# COINCIDENCES = ["all", "coincident", "noncoincident"]
COINCIDENCES = ["all"]

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
    if flare_label in "B":
        return 0
    elif flare_label in "C":
        return 1
    elif flare_label in "M":
        return 2
    elif flare_label in "X":
        return 3


def main():
    time_range = 24

    abc_properties_df = pd.read_csv("Data_ABC_with_Korsos_parms.csv")
    mx_properties_df = pd.read_csv("Data_MX_with_Korsos_parms.csv")

    correct_predictions = {k: 0 for k in FLARE_LABELS}
    incorrect_predictions = {k: 0 for k in FLARE_LABELS}

    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    # 2014 dataset
    root_directory = "classifiers/"
    data_directory = f"{root_directory}data_2014/"

    for is_coincident in COINCIDENCES:
        mean_df = pd.read_csv(f"{root_directory}mean_{is_coincident}.csv")

        bc_info_df = pd.read_csv(f"{data_directory}2014_{is_coincident}_bc.csv")
        mx_info_df = pd.read_csv(f"{data_directory}2014_{is_coincident}_mx.csv")
        info_df = pd.concat([bc_info_df, mx_info_df])
        info_df.reset_index(inplace=True)
        info_df.drop(["index", "Unnamed: 0"], axis=1, inplace=True)
        for time_string in ["time_start", "time_peak", "time_end"]:
            info_df[time_string] = \
                info_df[time_string].apply(parse_tai_string)

        df_needed = pd.DataFrame(columns=FLARE_PROPERTIES)
        for label in FLARE_LABELS:
            temp_df = info_df
            info_df = info_df.loc[info_df["xray_class"] == label]
            info_df.reset_index(inplace=True)
            for flare_index, row in info_df.iterrows():
                print(label, flare_index, "/", info_df.shape[0])
                # Find NOAA AR number and timestamp from user input in info dataframe.
                noaa_ar = row["nar"]
                timestamp = floor_minute(row["time_start"])
                start_time = row["time_start"]
                flare_class = row["xray_class"]
                coincidence = row["is_coincident"]

                if flare_class in ["B", "C"]:
                    properties_df = abc_properties_df
                else:
                    properties_df = mx_properties_df

                # Find corresponding ending index in properties dataframe.
                end_series = properties_df.loc[properties_df["T_REC"] == timestamp]
                if end_series.empty or (
                        end_series.loc[end_series['NOAA_AR'] == noaa_ar]).empty:
                    continue
                end_index = \
                    end_series.loc[end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]

                # Find corresponding starting index in properties dataframe, if it exists.
                start_index = end_index
                for i in range(time_range * 5 - 1):
                    if end_index - i >= 0:
                        if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
                            start_index = end_index - i

                # Make sub-dataframe of this flare
                local_properties_df = properties_df.iloc[start_index:end_index + 1]
                # local_properties_df.loc[:, 'xray_class'] = flare_class
                # local_properties_df.loc[:, 'time_start'] = start_time
                # local_properties_df.loc[:, 'is_coincident'] = coincidence
                # local_properties_df.loc[:, 'flare_index'] = flare_index
                df_needed = pd.concat([df_needed, local_properties_df])

                df = pd.DataFrame(columns=FLARE_PROPERTIES)
                for flare_property in FLARE_PROPERTIES:
                   df[flare_property] = [df_needed[flare_property].mean()]
                print(df)
                print(mean_df)
                exit(1)

            info_df = temp_df


if __name__ == "__main__":
    main()