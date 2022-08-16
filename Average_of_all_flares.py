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

FLARE_PROPERTIES += ['d_l_f', 'g_s', 'slf']

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
    if type(time) is str:
        time = parse_tai_string(time)
    return time - datetime.timedelta(
        minutes=time.minute % cadence) - datetime.timedelta(minutes=cadence)


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

    abc_properties = pd.read_csv("Data_ABC_with_Korsos_parms.csv")
    abc_properties.dropna(inplace=True)
    mx_properties = pd.read_csv("Data_MX_with_Korsos_parms.csv")
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

    labels = ["BC", "MX"]
    colors = ["blue", "red"]

    # labels = ["B", "C", "M", "X"]
    # colors = ["blue", "green", "orange", "red"]

    # labels = ["X"]
    # colors = ["red"]

    # labels = [flare.upper()]
    # colors = ["cyan"]


    # info_df = coincident_info_df
    # info_df = noncoincident_info_df

    # is_coincident = "all"
    # is_coincident = "coincident"
    # is_coincident = "noncoincident"

    idealized_flares = "idealized_flares2"

    min_max_df = pd.DataFrame()

    print(info_df.shape)
    print(coincident_info_df.shape)
    print(noncoincident_info_df.shape)

    def plot_idealized_time_series(is_coincident: str):
        fig, ax = plt.subplots(7, 3, figsize=(20, 22))
        min_max_csv = f"{idealized_flares}/{is_coincident}/24_average_flares_min_max_{is_coincident}.csv"
        for label, color in zip(labels, colors):
            csv = f"{idealized_flares}/{is_coincident}/{label}_idealized_flare.csv"
            df_ave = pd.read_csv(csv)

            # # Uncomment this for normalized data.
            min_max_df = pd.read_csv(min_max_csv)
            for flare_property in FLARE_PROPERTIES:
                x = df_ave[flare_property]
                maximum = min_max_df[flare_property][1]
                minimum = min_max_df[flare_property][0]
                df_ave[flare_property] = (x - minimum) / (maximum - minimum)
            # print(df_ave)

            # Plot specified flare properties over the specified time.
            row, col = 0, 0
            df_ave.dropna(inplace=True)
            df_ave.drop("Unnamed: 0", axis=1, inplace=True)

            for flare_property in FLARE_PROPERTIES:
                property_df = df_ave[[flare_property]]
                property_np = property_df.to_numpy().ravel()
                std_error = np.std(property_np, ddof=0) / np.sqrt(len(property_np))
                # property_df.plot(y=flare_property, ax=ax[row, col], color=color,
                #                  label=label)
                ax[row, col].errorbar(x=range(len(property_np)), y=property_np,
                                      yerr=std_error, capsize=4, color=color, label=label)
                ax[row, col].set_ylabel(flare_property)
                ax[row, col].set_title(f"{flare_property}")
                ax[row, col].legend()

                col += 1
                if col == 3:
                    col = 0
                    row += 1

        fig.tight_layout()
        # fig.savefig(f'{idealized_flares}/{is_coincident}/24h_idealized_flare_raw_errorbar_corrected.png')
        plt.savefig(f'{idealized_flares}/{is_coincident}/24h_idealized_flare_global_normalization_errorbar_corrected.png')
        fig.show()

    def generate_idealized_time_series(is_coincident, info_df):
        average_dfs = pd.DataFrame()
        min_max_csv = f"{idealized_flares}/{is_coincident}/24_average_flares_min_max_{is_coincident}.csv"

        df_needed = pd.DataFrame(columns=FLARE_PROPERTIES)
        df_1_sum = pd.DataFrame(columns=FLARE_PROPERTIES)
        df_2_sum = pd.DataFrame(columns=FLARE_PROPERTIES)
        for flare_property in FLARE_PROPERTIES:
            df_1_sum[flare_property] = np.zeros(time_range * 5)
            df_2_sum[flare_property] = np.zeros(time_range * 5)

        for label in labels:
            temp_df = info_df
            info_df = pd.concat([
                                    info_df.loc[info_df["xray_class"] == label[0]],
                                    info_df.loc[info_df["xray_class"] == label[1]]
            ])
            info_df.reset_index(inplace=True)
            for flare_index, row in info_df.iterrows():
                print(label, flare_index, "/", info_df.shape[0])
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

                df_1 = pd.DataFrame(columns=FLARE_PROPERTIES)
                df_2 = pd.DataFrame(columns=FLARE_PROPERTIES)
                for flare_property in FLARE_PROPERTIES:
                    df_1[flare_property] = np.zeros(time_range * 5)
                    df_2[flare_property] = np.zeros(time_range * 5)
                    for i in range(time_range * 5 - 1, -1, -1):
                        local_df_ind = end_index - (time_range * 5 - 1 - i)
                        if local_df_ind >= 0 and local_df_ind >= start_index:
                            df_1.at[i, flare_property] = local_properties_df.at[
                                local_df_ind, flare_property]
                        if df_1.at[i, flare_property] != 0:
                            df_2.at[i, flare_property] = 1

                local_properties_df.loc[:, 'xray_class'] = flare_class
                local_properties_df.loc[:, 'time_start'] = start_time
                # local_properties_df.loc[:, 'flare_index'] = flare_index
                df_needed = pd.concat([df_needed, local_properties_df])

                df_1_sum = df_1_sum.add(df_1)
                df_2_sum = df_2_sum.add(df_2)

            df_ave = df_1_sum.div(df_2_sum)
            df_ave.to_csv(f"{idealized_flares}/{is_coincident}/{label}_idealized_flare.csv")
            average_dfs = pd.concat([average_dfs, df_ave])
            info_df = temp_df

        # Below generates the CSV data for min/maxes.
        mins, maxes = [], []
        min_max_df = pd.DataFrame(columns=FLARE_PROPERTIES)
        for flare_property in FLARE_PROPERTIES:
            x = average_dfs[flare_property]
            mins.append(x.min())
            maxes.append(x.max())
        min_max_df.loc[len(min_max_df.index)] = mins
        min_max_df.loc[len(min_max_df.index)] = maxes
        min_max_df.to_csv(min_max_csv)

    for is_coincident in ["all", "coincident", "noncoincident"]:
        generate_idealized_time_series(is_coincident, info_df)
        # plot_idealized_time_series(is_coincident)


if __name__ == "__main__":
    main()
