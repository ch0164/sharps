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
from sklearn.metrics import classification_report
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
FLARE_PROPERTIES += ['d_l_f', 'g_s', 'slf']

FLARE_LABELS = ["B", "C", "M", "X"]
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


time_range = 12
abc_properties_df = pd.read_csv("Data_ABC_with_Korsos_parms.csv")
mx_properties_df = pd.read_csv("Data_MX_with_Korsos_parms.csv")

def main():

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
            timestamp = floor_minute(row["time_start"]) - datetime.timedelta(0, 12 * 60 * 60)
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
            else:
                end_index = \
                    end_series.loc[end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]


            # Find corresponding starting index in properties dataframe, if it exists.
            # start_index = end_index
            # for i in range(time_range * 5 - 1):
            #     if end_index - i >= 0:
            #         if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
            #             start_index = end_index - i

            # Make sub-dataframe of this flare
            flare = properties_df.iloc[end_index]
            # local_properties_df = properties_df.iloc[start_index:end_index + 1]
            new_df = pd.DataFrame(columns=FLARE_PROPERTIES)
            for flare_property in FLARE_PROPERTIES:
                new_df[flare_property] = [flare[flare_property].mean()]
            new_df['xray_class'] = [flare_class]
            new_df['is_coincident'] = [coincidence]
            df_needed = pd.concat([df_needed, new_df])

        # info_df = temp_df
        df_needed.reset_index(inplace=True)
        df_needed.drop(["index"], axis=1, inplace=True)
        coin_df = df_needed.loc[df_needed["is_coincident"] == True]
        noncoin_df = df_needed.loc[df_needed["is_coincident"] == False]
        df_needed.to_csv(f"pca2/singh_prime_flare_data_12h_timepoint_{is_coincident}.csv")
        coin_df.to_csv(f"pca2/singh_prime_flare_data_12h_timepoint_{'coincident'}.csv")
        noncoin_df.to_csv(f"pca2/singh_prime_flare_data_12h_timepoint_{'noncoincident'}.csv")


def generate_data(coincidence):
    df = pd.read_csv(f"singh_prime_flare_info_{coincidence}.csv")
    for time_string in ["time_start", "time_peak", "time_end"]:
        df[time_string] = \
            df[time_string].apply(parse_tai_string)

    df_needed = pd.DataFrame()
    for flare_index, row in df.iterrows():
        noaa_ar = row["nar"]
        timestamp = floor_minute(row["time_start"])
        flare_class = row["xray_class"]

        if flare_class in ["B", "C"]:
            properties_df = abc_properties_df
        else:
            properties_df = mx_properties_df

        end_series = properties_df.loc[properties_df["T_REC"] == timestamp]
        print(timestamp)
        print(end_series)
        end_index = \
            end_series.loc[end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]
        start_index = end_index
        for i in range(time_range * 5 - 1):
            if end_index - i >= 0:
                if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
                    start_index = end_index - i

        # Make sub-dataframe of this flare
        local_properties_df = properties_df.iloc[start_index:end_index + 1]
        local_properties_df.loc[:, 'xray_class'] = flare_class
        # local_properties_df.loc[:, 'is_coincident'] = coincidence
        new_df = pd.DataFrame(columns=FLARE_PROPERTIES)
        for flare_property in FLARE_PROPERTIES:
            new_df[flare_property] = local_properties_df[flare_property].mean()
        print(new_df)
        df_needed = pd.concat([new_df, df_needed])
        print(df_needed)
        exit(1)
    print(df)


def plot_reduced_scatter_matrix(df, coincidence):
    n_components = 4
    target = df["xray_class"]
    df = df.drop("xray_class", axis=1)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)

    total_var = pca.explained_variance_ratio_.sum() * 100
    print(pca.explained_variance_ratio_)

    labels = {str(i): f"PC {i + 1} ({var:.1f})" for i, var in zip(range(n_components), pca.explained_variance_ratio_ * 100)}
    labels['color'] = 'X-Ray Class'

    fig = px.scatter_matrix(
        components,
        color=target,
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}% ({coincidence.capitalize()}, 12h Timepoint Before Flare)',
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html(f"pca2/{coincidence}/12h_timepoint/reduced_scatter_matrix.html")


def plot_scatter_3d(df, coincidence):
    target = df["xray_class"]
    df = df.drop("xray_class", axis=1)

    n_components = 4
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)

    pc_labels = [f"PC {i+1}" for i in range(4)]

    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=target,
        title=f'Total Explained Variance: {total_var:.2f}% ({coincidence.capitalize()}, 12h Timepoint Before Flare)',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )

    fig.write_html(f"pca2/{coincidence}/12h_timepoint/pca_3d.html")



def plot_scatter_matrix(data, coincidence):
    fig = px.scatter_matrix(
        data,
        dimensions=FLARE_PROPERTIES,
        color="xray_class",
        width=2000,
        height=2000,
        title=f"({coincidence.capitalize()}, 12h Timepoint Before Flare)"
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html(f"pca2/{coincidence}/12h_timepoint/scatter_matrix.html")



if __name__ == "__main__":
    for coincidence in COINCIDENCES:
        # df = generate_data(coincidence)
        df = pd.read_csv(f"pca2/singh_prime_flare_data_12h_timepoint_{coincidence}.csv")
        df = df.drop(["is_coincident", "Unnamed: 0"], axis=1)
        plot_reduced_scatter_matrix(df, coincidence)
        plot_scatter_matrix(df, coincidence)
        plot_scatter_3d(df, coincidence)
    # main()