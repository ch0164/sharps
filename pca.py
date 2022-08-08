import datetime
from datetime import datetime as dt_obj

import pandas
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
import os

from sklearn.metrics import classification_report

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
# COINCIDENCES = ["all", "coincident", "noncoincident"]
COINCIDENCES = ["all"]


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


def classify(mean_df, std_df, param, x):
    m_mean = mean_df[param][2]
    m_std = std_df[param][2]
    c_mean = mean_df[param][1]
    c_std = std_df[param][1]

    new_mean1 = m_mean - m_std
    new_mean2 = c_mean + c_std
    new_mean = (new_mean1 + new_mean2) / 2

    if x < new_mean:  #  + (3 * m_std)
        return "ABC"
    else:
        return "MX"



def main():
    time_range = 24

    abc_properties_df = pd.read_csv("Data_ABC_with_Korsos_parms.csv")
    mx_properties_df = pd.read_csv("Data_MX_with_Korsos_parms.csv")

    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    info_df = pd.read_csv(f"all_flare_info_2.csv")
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    for is_coincident in COINCIDENCES:
        df_needed = pd.DataFrame(columns=FLARE_PROPERTIES)

        temp_df = info_df
        if is_coincident == "coincident":
            info_df = info_df.loc[info_df["is_coincident"] == True]
        elif is_coincident == "noncoincident":
            info_df = info_df.loc[info_df["is_coincident"] == False]

        b_df = info_df.loc[info_df["xray_class"] == "B"]
        c_df = info_df.loc[info_df["xray_class"] == "C"]
        m_df = info_df.loc[info_df["xray_class"] == "M"]
        x_df = info_df.loc[info_df["xray_class"] == "X"]

        df = pd.DataFrame()
        for flare_index, row in info_df.iterrows():
            print(flare_index, "/", info_df.shape[0])
            # Find NOAA AR number and timestamp from user input in info dataframe.
            noaa_ar = row["nar"]
            timestamp = floor_minute(row["time_start"])
            flare_class = row["xray_class"]

            if flare_class in ["B", "C"]:
                properties_df = abc_properties_df
            else:
                properties_df = mx_properties_df

            # Find corresponding ending index in properties dataframe.
            end_series = properties_df.loc[properties_df["T_REC"] == timestamp]
            if end_series.empty or (
                    end_series.loc[end_series['NOAA_AR'] == noaa_ar]).empty:
                continue
            else:
                df = df.append(row)
                print(df)
                continue
            #     end_index = \
            #         end_series.loc[end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]
            #
            # # Find corresponding starting index in properties dataframe, if it exists.
            # start_index = end_index
            # for i in range(time_range * 5 - 1):
            #     if end_index - i >= 0:
            #         if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
            #             start_index = end_index - i
            #
            # # Make sub-dataframe of this flare
            # local_properties_df = properties_df.iloc[start_index:end_index + 1]
            # local_properties_df.loc[:, 'xray_class'] = flare_class
            # local_properties_df.loc[:, 'is_coincident'] = is_coincident
            # new_df = pd.DataFrame(columns=FLARE_PROPERTIES)
            # for flare_property in FLARE_PROPERTIES:
            #     new_df[flare_property] = local_properties_df[flare_property].mean()
            # print(new_df)
            # df_needed = pd.concat([new_df, df_needed])
            # exit(1)

        # info_df = temp_df
        df.reset_index(inplace=True)
        df.drop(["Unnamed: 0", "index"], axis=1, inplace=True)
        coin_df = df.loc[df["is_coincident"] == True]
        noncoin_df = df.loc[df["is_coincident"] == False]
        df.to_csv(f"singh_prime_flare_info_{is_coincident}.csv")
        coin_df.to_csv(f"singh_prime_flare_info_{'coincident'}.csv")
        noncoin_df.to_csv(f"singh_prime_flare_info_{'noncoincident'}.csv")


if __name__ == "__main__":
    main()