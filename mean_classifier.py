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

FLARE_LABELS = ["B", "C", "M", "X"]
COINCIDENCES = ["all", "coincident", "noncoincident"]
# COINCIDENCES = ["all"]

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
    flare_property = "R_VALUE"

    abc_properties_df = pd.read_csv("Data_ABC_with_Korsos_parms.csv")
    mx_properties_df = pd.read_csv("Data_MX_with_Korsos_parms.csv")

    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    # 2014 dataset
    root_directory = "classifiers/"
    data_directory = f"{root_directory}data_2014/"

    info_df = pd.read_csv(f"{root_directory}2013_2014_flare_info.csv")
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    mean_df = pd.read_csv(f"{root_directory}mean_{'all'}.csv")
    std_df = pd.read_csv(f"{root_directory}std_{'all'}.csv")
    for is_coincident in COINCIDENCES:
        temp_df = info_df
        new_df = pd.DataFrame()
        if is_coincident == "coincident":
            info_df = info_df.loc[info_df["is_coincident"] == True]
        elif is_coincident == "noncoincident":
            info_df = info_df.loc[info_df["is_coincident"] == False]
        accuracies = pd.DataFrame(columns=FLARE_PROPERTIES)
        for flare_property in FLARE_PROPERTIES:
            y_true, y_pred = [], []
            flare_conf = np.zeros((2, 2), dtype=int)
            for flare_index, row in info_df.iterrows():
                df_needed = pd.DataFrame(columns=FLARE_PROPERTIES)
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
                df_needed = local_properties_df

                mean = df_needed[flare_property].mean()
                pred_class = classify(mean_df, std_df, flare_property, mean)
                if flare_class in "ABC":
                    y = "ABC"
                else:
                    y = "MX"
                y_true.append(y)
                y_pred.append(pred_class)
                flare_class_num, pred_class_num = flare_to_num(flare_class), flare_to_num(pred_class)
                if pred_class == flare_class:
                    flare_conf[flare_class_num][flare_class_num] += 1
                else:
                    flare_conf[flare_class_num][pred_class_num] += 1
            print(flare_conf)
            sns.heatmap(flare_conf, annot=True, cmap="Blues", cbar=False,
                        fmt="d",square=True,
                        xticklabels=["ABC", "MX"], yticklabels=["ABC", "MX"])
            plt.title(f"Mean-based Prediction on {flare_property} for "
                      f"{is_coincident.capitalize()} Flares")
            plt.tight_layout()
            plt.savefig(f"{root_directory}{is_coincident}/{flare_property}_{is_coincident}_mean.jpeg")
            plt.show()

            correct = flare_conf[0][0] + flare_conf[1][1]
            incorrect = flare_conf[0][1] + flare_conf[1][0]
            total = correct + incorrect
            accuracies[flare_property] = [correct / total]

            cr = classification_report(y_true, y_pred, target_names=["ABC", "MX"], output_dict=True)
            df = pandas.DataFrame(cr).transpose()
            df.to_csv(f"{root_directory}{is_coincident}/classification_reports/{flare_property}_classification_report_{is_coincident}.csv")

        accuracies.to_csv(f"{root_directory}{is_coincident}/accuracies_{is_coincident}.csv")
        info_df = temp_df


if __name__ == "__main__":
    main()