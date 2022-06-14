import datetime
from datetime import datetime as dt_obj
import matplotlib.pylab as plt
import pandas as pd
from scipy.stats import entropy
import seaborn as sns

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


def floor_minute(time, cadence=12):
    return time - datetime.timedelta(minutes=time.minute % cadence)


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
    # bc_info_df = pd.read_csv("../../Downloads/ABC_list.txt")
    # bc_properties_df = pd.read_csv("Data_ABC.csv")
    bc_info_df = None
    bc_properties_df = None
    # MX Flares
    mx_info_df = pd.read_csv("MX_list.txt")
    mx_properties_df = pd.read_csv("Data_MX.csv")
    # mx_info_df = None
    # mx_properties_df = None

    flare_class = "X"

    # info_df = pd.concat([bc_info_df, mx_info_df])
    # info_df.reset_index(inplace=True)
    # info_df.drop("index", axis=1, inplace=True)
    properties_df = pd.concat([bc_properties_df, mx_properties_df])
    properties_df.reset_index(inplace=True)
    properties_df.drop("index", axis=1, inplace=True)

    info_df = pd.read_csv("all_flares.txt")

    info_df["xray_class"] = info_df["xray_class"].apply(classify_flare)
    # info_df = info_df.loc[info_df["xray_class"] == "X"]

    # Define input for flare.
    time_range = 24  # Valid: 1 to 48 hours

    # Convert time strings to datetime objects for cleaned info data.
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    # Convert T_REC string to datetime objects.
    properties_df["T_REC"] = \
        properties_df["T_REC"].apply(parse_tai_string)

    linestyles = ["-", "--", ":", "None"]
    markers = ["s", "p", "P", "*", ".", "D"]
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    styles = [(ls, m, c) for ls in linestyles for m in markers for c in colors]

    dataframes = []
    series_df = pd.DataFrame()
    for flare_index, row in info_df.iterrows():
        print(flare_index, info_df.shape[0])
        # Find NOAA AR number and timestamp from user input in info dataframe.
        noaa_ar = info_df["nar"][flare_index]
        timestamp = floor_minute(info_df["time_start"][flare_index])

        # Find corresponding ending index in properties dataframe.
        abc_end_series = properties_df.iloc[
            (properties_df['T_REC'] - timestamp).abs().argsort()[:1]]
        abc_end_index = abc_end_series.index.tolist()[0]

        # Find corresponding starting index in properties dataframe, if it exists.
        abc_start_index = abc_end_index
        for i in range(time_range * 5 - 1):
            if abc_end_index - i >= 0:
                if properties_df["NOAA_AR"][abc_end_index - i] == noaa_ar:
                    abc_start_index = abc_end_index - i

        range_df = properties_df[abc_start_index:abc_end_index].drop(
            ["T_REC", "NOAA_AR"], axis=1)  # ["T_REC", "NOAA_AR"]
        # dataframes.append((range_df, row["xray_class"]))
        mean_df = range_df.mean().to_frame()

        series_df = pd.concat([
            series_df, mean_df.T])

    series_df.dropna(inplace=True)
    series_df.reset_index(inplace=True)
    series_df.drop("index", axis=1, inplace=True)
    print(series_df)

    # Plot specified flare properties over the specified time.
    # fig, ax = plt.subplots(6, 3, figsize=(18, 20))
    # fig.suptitle(f"{flare_class} Flares (Mean {time_range}h Time Series)", fontsize=24)
    # row, col = 0, 0
    # i = 0
    # for flare_property in FLARE_PROPERTIES:
    #     i += 1
    #     # print(f"{i}/{len(FLARE_PROPERTIES)}")
    #     print(flare_property, "&", series_df[flare_property].value_counts(), r"\\")
    #     if row == 0 and col == 1:
    #         col = 2
    #
    #     # for df, flare_class in dataframes:
    #     #     if "B" in flare_class:
    #     #         color = "c"
    #     #     elif "C" in flare_class:
    #     #         color = "g"
    #     #     elif "M" in flare_class:
    #     #         color = "orange"
    #     #     else:
    #     #         color = "r"
    #     #     ax[row, col].plot(range(series_df.shape[0]), series_df[flare_property])
    #
    #
    #
    #     ax[row, col].plot(range(len(series_df[flare_property])),
    #                       series_df[flare_property])
    #     ax[row, col].set_ylabel(flare_property)
    #     ax[row, col].set_title(flare_property)
    #
    #     col += 1
    #     if col == 3:
    #         col = 0
    #         row += 1
    #
    # fig.tight_layout()
    # fig.show()
    # fig.savefig(f"time_series/{time_range}h/{flare_class.lower()}_flares_mean")
    df = series_df.loc[:, FLARE_PROPERTIES]
    cm = df.corr()
    print(cm)
    # sns.heatmap(cm, annot=True, cmap="Blues", cbar=False, fmt="d",
    #             square=True, xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    # plt.title(f"{flare_class} Flares (Mean {time_range}h Time Series)")
    # plt.show()

if __name__ == "__main__":
    main()
