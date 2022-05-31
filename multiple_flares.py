import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib

import matplotlib
from astropy.io import fits
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import seaborn as sns
import sunpy.map
import sunpy.io
from IPython.display import Image
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import csv
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
    # plt.style.use("dark_background")
    # Choose which flares to plot.
    # ABC Flares
    # abc_info_df = pd.read_csv("ABC_list.txt")
    abc_properties_df = pd.read_csv("Data_ABC.csv")
    # MX Flares
    # mx_info_df = pd.read_csv("MX_list.txt")
    mx_properties_df = pd.read_csv("Data_MX.csv")

    info_df = pd.read_csv("all_flares.txt")
    info_df.drop(["hec_id", "lat_hg", "long_hg", "long_carr", "optical_class"], axis=1, inplace=True)

    # Convert time strings to datetime objects for cleaned info data.
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    # Label flares by B, C, M, and X.
    info_df["xray_class"] = \
        info_df["xray_class"].apply(classify_flare)

    # Get B and C class flares, round down their minutes.
    b_df = info_df.loc[info_df["xray_class"] == "B"]
    c_df = info_df.loc[info_df["xray_class"] == "C"]

    bc_info = pd.concat([b_df, c_df])
    bc_info["time_start"] = \
        bc_info["time_start"].apply(floor_minute)

    # Find the respective timestamp in the ABC data file.
    bc_data = pd.DataFrame()
    for index, row in bc_info.iterrows():
        timestamp = row["time_start"]
        bc_series = abc_properties_df.loc[
            abc_properties_df["T_REC"] == timestamp].head(1)
        # bc_series.insert(0, "CLASS", row["xray_class"])
        bc_data = pd.concat([bc_data, bc_series])

    # Get M and X class flares, round down their minutes.
    m_df = info_df.loc[info_df["xray_class"] == "M"]
    x_df = info_df.loc[info_df["xray_class"] == "X"]
    mx_info = pd.concat([m_df, x_df])
    mx_info["time_start"] = \
        mx_info["time_start"].apply(floor_minute)

    # Find the respective timestamp in the MX data file.
    mx_data = pd.DataFrame()
    for index, row in mx_info.iterrows():
        timestamp = row["time_start"]
        mx_series = mx_properties_df.loc[
            abc_properties_df["T_REC"] == timestamp].head(1)
        # mx_series.insert(0, "CLASS", row["xray_class"])
        mx_data = pd.concat([mx_data, mx_series])

    # Find timepoints where multiple flares occur in complete set,
    # regardless of flare class.
    # Note: Complete set is sorted by flare class, starting from B, C, M, and X.
    # 1. Take the end time of the event to be the end time of the actual flare.
    # 2. Take the start time of the event to be 24 hours prior to the end time.
    # 3. For all other flares in the complete set:
    #    i. Get the other flare's range like above, then determine if both
    #    flares' ranges have any overlap -- if so, then append it to a list.

    flare_info = pd.concat([mx_info, bc_info], ignore_index=True)
    # flare_info = pd.concat([mx_info], ignore_index=True)
    # flare_info.drop("Unnamed: 0", axis=1, inplace=True)

    b_df = flare_info.loc[flare_info["xray_class"] == "B"]
    c_df = flare_info.loc[flare_info["xray_class"] == "C"]
    m_df = flare_info.loc[flare_info["xray_class"] == "M"]
    x_df = flare_info.loc[flare_info["xray_class"] == "X"]

    print("B Class Flares Size:", b_df.shape)
    print("C Class Flares Size:", c_df.shape)
    print("M Class Flares Size:", m_df.shape)
    print("X Class Flares Size:", x_df.shape)

    text = f"""Class B Flares Shape: {b_df.shape}
Class C Flares Shape: {c_df.shape}
Class M Flares Shape: {m_df.shape}
Class X Flares Shape: {x_df.shape}"""

    # MULTIPLE FLARES BELOW
    flare_info = pd.concat([b_df, c_df, m_df, x_df])
    flare_info.reset_index(inplace=True)
    print(flare_info)
    # flare_matrix = np.zeros(shape=(flare_info.shape[0], flare_info.shape[0]),
    #                         dtype=int)
    #
    # flare_conf = np.zeros(shape=(5, 5), dtype=int)

    def class_to_num(flare_class):
        if flare_class == "B":
            return 0
        elif flare_class == "C":
            return 1
        elif flare_class == "M":
            return 2
        else:
            return 3

    # flare_coincidences_mix = [0 for _ in range(flare_info.shape[0])]
    # flare_coincidences_same = [0 for _ in range(flare_info.shape[0])]
    # print(len(flare_coincidences_same))
    #
    # for index1, row1 in flare_info.iterrows():
    #     flare_class1 = row1["xray_class"]
    #     print(index1)
    #     time_end1 = row1["time_end"]
    #     time_start1 = time_end1 - datetime.timedelta(1)
    #     for index2, row2 in flare_info.iterrows():
    #         flare_class2 = row2["xray_class"]
    #         # Don't count the same flare.
    #         if index1 == index2:
    #             continue
    #         # Only look for flares in the same class.
    #         time_end2 = row2["time_end"]
    #         time_start2 = time_end2 - datetime.timedelta(1)
    #         flares_overlap = (time_start1 <= time_start2 <= time_end1) or (
    #                 time_start1 <= time_end2 <= time_end1)
    #         if flares_overlap:
    #             # if flare_class1 == flare_class2:
    #             #     # if flare_class1 == "C":
    #             #     #     flares_overlap
    #             #     # elif flare_class1 == "M":
    #             #     #     flare_matrix[index1][index2] = 1
    #             #     # elif flare_class1 == "B":
    #             #     #     flare_matrix[index1][index2] = 2
    #             #     flare_coincidences_same[index1] += 1
    #             # else:
    #             #     flare_matrix[index1][index2] = 3
    #
    #             flare_coincidences_mix[index1] += 1

    def plot_examples(data, colors):
        """
        Helper function to plot data with associated colormap.
        """
        colormaps = ListedColormap(colors)
        plt.figure(figsize=(16, 9))
        im = plt.imshow(data, interpolation='nearest', cmap=colormaps)
        plt.title("Coinciding Flares (M and B Flares)")
        labels = ["None", "M-M", "B-B", "M-B"]
        patches = [matplotlib.patches.Patch(color=colors[i],
                                            label=labels[i]) for i in
                   range(len(labels))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)
        plt.show()

    # colors = ["white", "purple", "red", "green"]
    # plot_examples(np.triu(flare_matrix).transpose(), colors)
    #
    # im = plt.imshow(flare_conf, interpolation="nearest")
    # for (j, i), label in np.ndenumerate(flare_conf):
    #     plt.text(i, j, label, ha='center', va='center', color="black")
    #     plt.text(i, j, label, ha='center', va='center', color="black")
    # plt.colorbar(im)
    # sns.heatmap(flare_conf, annot=True, cmap="Blues", cbar=False, fmt="d",
    #             square=True, xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    # plt.title("Flare Coincidence Confusion Matrix")
    # plt.show()
    # exit(1)

    #
    # fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    # mix_yticks = range(max(flare_coincidences_mix) + 1)
    # same_yticks = range(max(flare_coincidences_same) + 1)
    # ax[0].set_title(f"Coinciding Flares (M and B Flares Mix)")
    # ax[0].set_xlabel("Flare Index")
    # ax[0].set_ylabel("Number of Coincidental Flares")
    # ax[0].set_yticks(mix_yticks, mix_yticks)
    # # ax[1].text(400, max(flare_coincidences_mix) - 4, text)
    # ax[0].plot(range(flare_info.shape[0]), flare_coincidences_mix, color="y")
    # ax[1].set_title(f"Coinciding Flares (M and B Flares Same)")
    # ax[1].set_xlabel("Flare Index")
    # ax[1].set_ylabel("Number of Coincidental Flares")
    # ax[1].set_yticks(same_yticks, same_yticks)
    # ax[1].plot(range(flare_info.shape[0]), flare_coincidences_same, color="y")
    # fig.tight_layout()
    # fig.show()

    def info_to_data(info, data):
        df = pd.DataFrame()
        for index, row in info.iterrows():
            # if flare_coincidences_mix[index] <= 0:
            # if flare_coincidences_mix[index] > 0:
            #     continue
            timestamp = row["time_start"]
            df_sort = data.iloc[
                (data['T_REC'] - timestamp).abs().argsort()[:1]]
            df_sort.insert(0, "xray_class", row["xray_class"])
            # df_sort["xray_class"] = row["xray_class"]
            df = pd.concat([df, df_sort])
        return df

    b_data_df = info_to_data(b_df, bc_data)
    c_data_df = info_to_data(c_df, bc_data)
    m_data_df = info_to_data(m_df, mx_data)
    x_data_df = info_to_data(x_df, mx_data)
    flare_dataframes = [b_data_df, c_data_df, m_data_df, x_data_df]

    # Plot PCA
    # data_df = info_to_data(info_df, pd.concat([bc_data, mx_data]))
    # data_df.reset_index(inplace=True)
    # print(data_df, data_df.columns)
    # n = 6
    # pc_labels = [f"PC{i}" for i in range(1, n + 1)]
    # pca = PCA(n_components=n)
    # flare_pca = data_df.drop(["xray_class", "T_REC", "NOAA_AR"], axis=1)
    # flare_pca = pca.fit_transform(MinMaxScaler().fit_transform(flare_pca))
    # pca_df = pd.DataFrame(data=flare_pca, columns=pc_labels)
    # print(pca_df, pca_df.columns)
    # pca_df["xray_class"] = pd.Series(data_df["xray_class"])
    #
    # df = pca_df
    #
    # # df = pd.concat([pca_df, data_df["xray_class"]], axis=1, ignore_index=True)
    # ev = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2]
    # fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="xray_class", title=f"PCA (All Flares)\nTotal Explained Variance: {ev}")
    # fig.write_html(f"pca/all_flares/pca_3d.html")


    for label, flare_df in zip(CLASS_LABELS, flare_dataframes):
        n = 6
        pc_labels = [f"PC{i}" for i in range(1, n + 1)]
        pca = PCA(n_components=n)
        flare_pca = flare_df.drop(["xray_class", "T_REC", "NOAA_AR"], axis=1)
        flare_pca = pca.fit_transform(MinMaxScaler().fit_transform(flare_pca))
        pca_df = pd.DataFrame(data=flare_pca, columns=pc_labels)

        print(pca_df, pca_df.columns)

        ev = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2] + pca.explained_variance_ratio_[3]

        fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="PC4", title=f"Class {label} PCA\nTotal Explained Variance: {ev}")
        fig.write_html(f"pca/all_flares/{label}_3d.html")






    # flare_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # fig, ax = plt.subplots(2, 2)
    # for df, label, indices in zip(flare_dataframes, CLASS_LABELS,
    #                               flare_indices):
    #     i, j = indices
        # df.drop(["T_REC", "NOAA_AR"], axis=1, inplace=True)
        # if label == "X":
        #     n = 13
        # else:
        #     n = 17
        # n = 6
        # pc_labels = [f"PC{i}" for i in range(1, n + 1)]
        # pca = PCA(n_components=n)
        # pca.fit_transform(MinMaxScaler().fit_transform(df))
        # ev = pca.explained_variance_ratio_
        # ax[i, j].set_title(f"Class {label} PCA ({df.shape[0]} Flares)")
        # ax[i, j].set_xlabel("Principal Components")
        # ax[i, j].set_ylabel("Explained Variance Ratio")
        # ax[i, j].set_xticks(range(n), pc_labels, fontsize=8,
        #                     rotation="vertical")
        # ax[i, j].bar(range(len(ev)), list(ev * 100),
        #              align="center", color="y")
        # r = np.abs(pca.components_.T)
        # r /= r.sum(axis=0)
        # r = r.transpose()
        #
        # pca_df = pd.DataFrame(r, columns=FLARE_PROPERTIES)
        # total = []
        # for property in FLARE_PROPERTIES:
        #     total.append(np.multiply(pca_df[property],
        #                              pca.explained_variance_ratio_).sum())
        # pca_df.loc["Total", :] = pca_df.sum(axis=0)
        # print(len(pca_df.loc["Total", :]), len(pca.components_))
        # pca_df.loc["Total", :].multiply(pca.components_)
        # pca_df.index = pc_labels + ["Total"]
        # print("TOTAL", len(total), total)
        # pca_df = pca_df.T
        # pca_df["Total"] = total
        # print("Flare Class", label)
        # print(pca_df.to_latex())
        # print("Flare Class", label, "Sum:", f"{pca_df['Total'].sum():.6f}")

        # pca_df = pd.DataFrame(pca.explained_variance_ratio_, pc_labels)
        # pd.set_option('display.float_format', '{:.6f}'.format)
        # print(pca_df.to_latex())

    # fig.tight_layout()
    # fig.show()



if __name__ == "__main__":
    main()
