import datetime
from datetime import datetime as dt_obj
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from scipy.stats import chisquare
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


def floor_minute(time, cadence=12):
    return time - datetime.timedelta(minutes=time.minute % cadence) - datetime.timedelta(minutes=cadence)


def classify_flare(magnitude):
    if "B" in magnitude:
        return "B"
    elif "C" in magnitude:
        return "C"
    elif "M" in magnitude:
        return "M"
    else:
        return "X"


def main():
    # Choose which flares to plot.
    # ABC Flares
    # info_df = pd.read_csv("ABC_list.txt")
    # properties_df = pd.read_csv("Data_ABC.csv")
    # MX Flares
    # info_df = pd.read_csv("MX_list.txt")
    # properties_df = pd.read_csv("Data_MX.csv")

    info_df = pd.read_csv("all_flare_info_2.csv")
    # info_df = pd.read_csv("noncoincident_flare_info.csv")

    abc_properties = pd.read_csv("Data_ABC.csv")
    abc_properties.dropna(inplace=True)
    mx_properties = pd.read_csv("Data_MX.csv")
    mx_properties.dropna(inplace=True)

    def info_to_data(info):
        df = pd.DataFrame()
        for index, row in info.iterrows():
            flare_class = row["xray_class"]
            if flare_class in ["B", "C"]:
                data = abc_properties
            else:
                data = mx_properties
            print(f"{index}/{info.shape[0]}")
            timestamp = row["time_start"] - datetime.timedelta(0, 3600 * 6)
            nar_df = data.loc[data["NOAA_AR"] == row["nar"]]
            df_sort = nar_df.iloc[
                (nar_df['T_REC'] - timestamp).abs().argsort()[:1]]
            df_sort.insert(0, "xray_class", row["xray_class"])
            df = pd.concat([df, df_sort])
        print(df)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        return df

    coincident_info_df = info_df.loc[info_df["is_coincident"] == True]
    noncoincident_info_df = info_df.loc[info_df["is_coincident"] == False]

    # Define input for flare.
    # flare_index = 20  # Valid: 0 to 765
    time_range = 24  # Valid: 1 to 48 hours

    for info_df, is_coincident in [(noncoincident_info_df, "Noncoincident")]:
        plt.clf()
        info_df.reset_index(inplace=True)
        info_df.drop("index", axis=1, inplace=True)
        info_df.drop("Unnamed: 0", axis=1, inplace=True)
        print(info_df)
        df_1_sum = pd.DataFrame(columns=FLARE_PROPERTIES)
        df_2_sum = pd.DataFrame(columns=FLARE_PROPERTIES)
        for flare_property in FLARE_PROPERTIES:
            df_1_sum[flare_property] = np.zeros(time_range * 5)
            df_2_sum[flare_property] = np.zeros(time_range * 5)

        # Convert time strings to datetime objects for cleaned info data.
        for time_string in ["time_start", "time_peak", "time_end"]:
            info_df[time_string] = \
                info_df[time_string].apply(parse_tai_string)

        # Convert T_REC string to datetime objects.
        abc_properties["T_REC"] = \
            abc_properties["T_REC"].apply(parse_tai_string)
        mx_properties["T_REC"] = \
            mx_properties["T_REC"].apply(parse_tai_string)
        df_needed = pd.DataFrame(columns=FLARE_PROPERTIES)

        # Label flares by B, C, M, and X.
        info_df["xray_class"] = \
            info_df["xray_class"].apply(classify_flare)

        labels = ["B", "C", "M", "X"]
        colors = ["blue", "green", "orange", "red"]

        # labels = ["X"]
        # colors = ["red"]

        # labels = ["B", "C"]
        # colors = ["cyan", "green"]

        fig, ax = plt.subplots(6, 3, figsize=(18, 20))

        print(info_df["xray_class"].value_counts())

        for label, color in zip(labels, colors):
            temp_df = info_df
            info_df = info_df.loc[info_df["xray_class"] == label]
            info_df.reset_index(inplace=True)
            for flare_index, row in info_df.iterrows():
                print(flare_index, "/", info_df.shape[0])
                # Find NOAA AR number and timestamp from user input in info dataframe.
                noaa_ar = row["nar"]
                timestamp = floor_minute(row["time_start"])
                start_time = row["time_start"]
                flare_class = row["xray_class"]

                if flare_class in ["B", "C"]:
                    properties_df = abc_properties
                else:
                    properties_df = mx_properties

                # Find corresponding ending index in properties dataframe.
                end_series = properties_df.loc[properties_df["T_REC"] == timestamp]
                if end_series.empty or (end_series.loc[end_series['NOAA_AR'] == noaa_ar]).empty:
                    continue
                end_index = end_series.loc[end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]

                # Find corresponding starting index in properties dataframe, if it exists.
                start_index = end_index
                for i in range(time_range * 5 - 1):
                    if end_index - i >= 0:
                        if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
                            start_index = end_index - i

                # Make sub-dataframe of this flare
                local_properties_df = properties_df.iloc[start_index:end_index + 1]

                df_1 = pd.DataFrame(columns=FLARE_PROPERTIES)
                df_2 = pd.DataFrame(columns=FLARE_PROPERTIES)
                for flare_property in FLARE_PROPERTIES:
                    df_1[flare_property] = np.zeros(time_range * 5)
                    df_2[flare_property] = np.zeros(time_range * 5)
                    for i in range(time_range * 5 - 1, -1, -1):
                        local_df_ind = end_index - (time_range * 5 - 1 - i)
                        if local_df_ind >= 0 and local_df_ind >= start_index:
                            df_1.at[i, flare_property] = local_properties_df.at[local_df_ind, flare_property]
                        if df_1.at[i, flare_property] != 0:
                            df_2.at[i, flare_property] = 1

                local_properties_df.loc[:, 'xray_class'] = flare_class
                local_properties_df.loc[:, 'time_start'] = start_time
                # local_properties_df.loc[:, 'flare_index'] = flare_index
                df_needed = pd.concat([df_needed, local_properties_df])

                df_1_sum = df_1_sum.add(df_1)
                df_2_sum = df_2_sum.add(df_2)

            # print(df_1_sum)
            # print(df_2_sum)
            # print(df_needed)
            # df_needed.to_csv('MX_data_bernard.csv')

            df_ave = df_1_sum.div(df_2_sum)
            for flare_property in FLARE_PROPERTIES:
                x = df_ave[flare_property]
                df_ave[flare_property] =(x - x.min())/ (x.max() - x.min())
            df_ave.to_csv(f"24_average_{label.lower()}_{is_coincident}.csv")

            info_df = temp_df

        # df_ave = pd.read_csv("24_average_x_all.csv")
        # print(df_ave.columns)
        # Plot specified flare properties over the specified time.
            row, col = 0, 0
            print(df_ave)
            for flare_property in FLARE_PROPERTIES:
                property_df = df_ave[[flare_property]]
                property_df.plot(y=flare_property, ax=ax[row, col], color=color, label=label)
                ax[row, col].set_ylabel(flare_property)
                # ax[row, col].set_title(f"Total {flare_property} from {properties_df['T_REC'].values[0]}")
                ax[row, col].set_title(f"{flare_property}")

                col += 1
                if col == 3:
                    col = 0
                    row += 1

            fig.tight_layout()
            # fig.legend(loc="lower right")
            fig.show()
            # plt.show()
        plt.savefig(f'idealized_time_series_normalized_{is_coincident}.png')


if __name__ == "__main__":
    main()
