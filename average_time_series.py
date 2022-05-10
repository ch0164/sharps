import datetime
from datetime import datetime as dt_obj
import matplotlib.pylab as plt
import pandas as pd

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


def floor_minute(time, cadence=12):
    return time - datetime.timedelta(minutes=time.minute % cadence)


def main():
    # Choose which flares to plot.
    # ABC Flares
    info_df = pd.read_csv("ABC_list.txt")
    properties_df = pd.read_csv("Data_ABC.csv")
    # MX Flares
    # info_df = pd.read_csv("MX_list.txt")
    # properties_df = pd.read_csv("Data_MX.csv")

    # Define input for flare.
    flare_index = 9  # Valid: 0 to 765
    time_range = 12  # Valid: 1 to 48 hours

    # Convert time strings to datetime objects for cleaned info data.
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    # Convert T_REC string to datetime objects.
    properties_df["T_REC"] = \
        properties_df["T_REC"].apply(parse_tai_string)

    # Find NOAA AR number and timestamp from user input in info dataframe.
    noaa_ar = info_df["nar"][flare_index]
    timestamp = floor_minute(info_df["time_start"][flare_index])

    # Find corresponding ending index in properties dataframe.
    abc_end_series = properties_df.loc[
        properties_df["T_REC"] == timestamp]
    abc_end_index = abc_end_series.loc[
        abc_end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]

    # Find corresponding starting index in properties dataframe, if it exists.
    abc_start_index = abc_end_index
    for i in range(time_range * 5 - 1):
        if abc_end_index - i >= 0:
            if properties_df["NOAA_AR"][abc_end_index - i] == noaa_ar:
                abc_start_index = abc_end_index - i

    # Plot specified flare properties over the specified time.
    fig, ax = plt.subplots(6, 3, figsize=(18, 20))
    row, col = 0, 0
    for flare_property in FLARE_PROPERTIES:
        property_df = properties_df[["T_REC", flare_property]]
        property_df.iloc[abc_start_index:abc_end_index].plot(
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


if __name__ == "__main__":
    main()
