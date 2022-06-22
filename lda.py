import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib

import matplotlib
from astropy.io import fits
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import seaborn as sns
import sunpy.map
import sunpy.io
import datetime as dt
from sklearn.linear_model import LinearRegression
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
    plt.style.use("dark_background")
    abc_properties_df = pd.read_csv("Data_ABC.csv")
    abc_properties_df.drop(to_drop, inplace=True, axis=1)
    mx_properties_df = pd.read_csv("Data_MX.csv")
    mx_properties_df.drop(to_drop, inplace=True, axis=1)

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

    info_df = info_df[(info_df["time_start"] >= pd.Timestamp(2010, 5, 1))]
    info_df.reset_index(inplace=True)
    info_df.drop("index", axis=1)

    b_df = info_df.loc[info_df["xray_class"] == "B"]
    c_df = info_df.loc[info_df["xray_class"] == "C"]
    m_df = info_df.loc[info_df["xray_class"] == "M"]
    x_df = info_df.loc[info_df["xray_class"] == "X"]

    single_dfs = [(x_df, "X"), (m_df, "M"), (b_df, "B"), (c_df, "C")]

    bc_info = pd.concat([
        b_df,
        c_df
    ])
    mx_info = pd.concat([
        m_df,
        x_df
    ])
    bx_info = pd.concat([
        b_df,
        x_df
    ])
    pair_dfs = [(mx_info, "MX"), (bx_info, "BX"), (bc_info, "BC")]

    # time_start_df = pd.read_csv("time_start_data.csv")
    # time_start_df.drop("Unnamed: 0", axis=1, inplace=True)

    series_df = pd.read_csv("series_data.csv")
    series_df.drop("Unnamed: 0", axis=1, inplace=True)
    # series_df = time_start_df
    # print(series_df)
    # exit(1)

    b_df = series_df.loc[series_df["xray_class"] == "B"]
    m_df = series_df.loc[series_df["xray_class"] == "M"]
    x_df = series_df.loc[series_df["xray_class"] == "X"]
    shape = b_df.shape[0] + m_df.shape[0] + x_df.shape[0]
    for df in [b_df, m_df, x_df]:
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
    print(b_df.to_string())
    print(m_df.to_string())
    # exit(1)

    df2 = pd.DataFrame()
    for df in [b_df, m_df, x_df]:
        X = df.drop("xray_class", axis=1)
        y = df["xray_class"]
        n = 7
        pc_labels = [f"PC{i}" for i in range(1, n + 1)]
        pca = PCA()
        flare_pca = pca.fit_transform(MinMaxScaler().fit_transform(X))
        ev = pca.explained_variance_ratio_
        total_ev = ev[0] + ev[1] + ev[2]
        pca_df = pd.DataFrame(data=flare_pca, columns=pc_labels)
        print(pca_df, pca_df.columns)
        pca_df["xray_class"] = pd.Series(y)
        df2 = pd.concat([pca_df, df2])

    pca_df = df2
    print(pca_df.to_string())
    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="xray_class", opacity=0.7,
                        title=f"PCA Non-Correlative Parameters for BMX ({shape} Flares)")
    fig.write_html(f"pca/non_correlative/24h/bmx_mean_3d.html")

    # ev = pca.explained_variance_ratio_
    # pcs = [f"PC{i}" for i in range(1, len(ev) + 1)]
    # plt.title(f"Non-Correlative Parameters PCA, All Flare Classes ({series_df.shape[0]} Flares)")
    # plt.xlabel("Principal Components")
    # plt.ylabel("Explained Variance Ratio")
    # plt.xticks(range(len(ev)), pcs, fontsize=8, rotation=30)
    # plt.bar(range(len(ev)), list(ev * 100),
    #         align="center", color="y")
    # plt.show()



    # n = 7
    # pc_labels = [f"PC{i}" for i in range(1, n + 1)]
    # pca = PCA()
    # X = series_df.drop("xray_class", axis=1)
    # y = series_df["xray_class"]
    # flare_pca = pca.fit_transform(MinMaxScaler().fit_transform(X))
    # ev = pca.explained_variance_ratio_
    # total_ev = ev[0] + ev[1] + ev[2]
    # pca_df = pd.DataFrame(data=flare_pca, columns=pc_labels)
    # print(pca_df, pca_df.columns)
    # pca_df["xray_class"] = pd.Series(y)
    #
    # print(pca_df)

    # colors = ["cyan", "lime", "orange", "red"]
    # fig = plt.figure(figsize=(25, 12))
    # pd.plotting.parallel_coordinates(pca_df,
    #                                  "xray_class", color=colors)
    # fig.tight_layout()
    # # fig.suptitle(f"{flare_class} Complete ({time_label})",
    # #              fontsize=20)
    # fig.show()
    # # fig.savefig(f"parallel_coordinates/{time_label}/complete_{flare_class}_class.png")
    #
    # exit(1)

    # fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="xray_class", opacity=0.7,
    #                     title=f"PCA Non-Correlative Parameters for All Class Flares ({series_df.shape[0]} Flares)"
    #                           f" Total EV: {total_ev}")
    # fig.write_html(f"pca/non_correlative/time_start/bcmx_mean_3d.html")
    #
    # ev = pca.explained_variance_ratio_
    # pcs = [f"PC{i}" for i in range(1, len(ev) + 1)]
    # plt.title(f"Non-Correlative Parameters PCA, All Flare Classes ({series_df.shape[0]} Flares)")
    # plt.xlabel("Principal Components")
    # plt.ylabel("Explained Variance Ratio")
    # plt.xticks(range(len(ev)), pcs, fontsize=8, rotation=30)
    # plt.bar(range(len(ev)), list(ev * 100),
    #         align="center", color="y")
    # plt.show()


    # fig = px.scatter_3d(series_df, x="PC1", y="PC2", z="PC3", color="MEANGBT",
    #                     symbol="xray_class", size="MEANGAM",
    #                     title=f"Non-Correlative Parameters for {label} Class Flares ({series_df.shape[0]} Flares)")
    # fig.write_html(f"correlation/{time_range_str}/{label}_mean_3d_3.html")

    # lda = LinearDiscriminantAnalysis()
    # X = time_start_df.drop("xray_class", axis=1)
    # y = time_start_df["xray_class"]
    # data_lda = lda.fit_transform(X.to_numpy(), y.to_numpy())

    # ev = lda.explained_variance_ratio_
    # iris_pc = [f"LD{i}" for i in range(1, len(ev) + 1)]
    # plt.title("Non-Correlative Parameters LDA, All Flare Classes")
    # plt.xlabel("Linear Discriminants")
    # plt.ylabel("Explained Variance Ratio")
    # plt.xticks(range(len(ev)), iris_pc, fontsize=8, rotation=30)
    # plt.bar(range(len(ev)), list(ev * 100),
    #           align="center", color="y")
    # plt.show()
    # exit(1)

    # data_lda = pd.DataFrame(data_lda, columns=["LD1", "LD2", "LD3"])


    # plt.figure()
    # colors = ['c', 'g', 'orange', "r"]
    # for color, i, target_name in zip(colors, CLASS_LABELS, CLASS_LABELS):
    #     plt.scatter(data_lda[y == i, 0], data_lda[y == i, 1], alpha=.8,
    #                 color=color,
    #                 label=target_name)
    #
    # fig = px.scatter_3d(data_lda, x="MEANJZD", y="MEANJZH", z="MEANGBT",
    #                     color="xray",
    #                     title=f"Non-Correlative Parameters for {label} Class Flares ({series_df.shape[0]} Flares)")
    # fig.write_html(f"bcmx_3d.html")

    # add legend to plot
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.xlabel("LD1")
    # plt.ylabel("LD2")
    # plt.title(f"time_start LDA on BCMX Non-Correlative Parameters ({time_start_df.shape[0]} Flares)")
    #
    # # display LDA plot
    # plt.show()

    # data_lda["xray_class"] = y
    # fig = px.scatter_3d(data_lda, x="LD1", y="LD2", z="LD3",
    #                     color="xray_class",
    #                     title=f"24h Range LDA on BCMX Non-Correlative Parameters ({time_start_df.shape[0]} Flares)",
    #                     opacity=0.7)
    # fig.write_html(f"lda.html")


if __name__ == "__main__":
    main()