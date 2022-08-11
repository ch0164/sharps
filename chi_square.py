import datetime

import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib

import matplotlib
from astropy.io import fits
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.decomposition import PCA
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
from scipy.stats import chisquare

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
    labels = ["B", "C", "M", "X"]
    n = len(FLARE_PROPERTIES)
    idealized_flares2 = "idealized_flares2"

    for is_coincident in ["all", "coincident", "noncoincident"]:
        # plt.clf()
        min_max_df = pd.read_csv(f"idealized_flares2/{is_coincident}/24_average_flares_min_max_{is_coincident}.csv")
        def generate_binned_csv():
            for label in labels:
                df = pd.read_csv(f"{idealized_flares2}/{is_coincident}/{label}_idealized_flare_{is_coincident}.csv")
                for flare_property in FLARE_PROPERTIES:
                    mn, mx = tuple(min_max_df[flare_property])
                    df[flare_property] = (df[flare_property] - mn) / (mx - mn)

                new_df = pd.DataFrame(columns=FLARE_PROPERTIES, index=range(24))
                bins = []
                for hour in range(0, 24):
                    if hour == 0:
                        bin_size = 4
                        increment = 1
                    else:
                        bin_size = 5
                        increment = 0
                    bin_ = [5 * hour + i + increment for i in range(0, bin_size)]
                    bins.append(bin_)

                for hour in range(0, 24):
                    bin_ = bins[hour]
                    data_points = df.iloc[bin_, :]
                    for flare_property in FLARE_PROPERTIES:
                        new_df[flare_property][hour] = data_points[
                            flare_property].mean()
                # for flare_property in FLARE_PROPERTIES:
                #     mn, mx = tuple(min_max_df[flare_property])
                #     new_df[flare_property] = (new_df[flare_property] - mn) / (mx - mn)
                print(new_df)
                new_df.to_csv(f"chi_square/{is_coincident}/{label}_idealized_flare_binned_{is_coincident}.csv")

        def generate_chi_square():
            for label in labels:
                df = pd.read_csv(f"chi_square/{is_coincident}/{label}_idealized_flare_binned_{is_coincident}.csv")
                x2 = np.zeros((n, n), dtype=float)

                def property_to_num(property):
                    return FLARE_PROPERTIES.index(property)

                for property1 in FLARE_PROPERTIES:
                    other_properties = sorted(list(set(FLARE_PROPERTIES) - {property1}))
                    print(property1)
                    print(other_properties)
                    for property2 in other_properties:
                        f_obs = df[property1]
                        f_exp = df[property2]
                        f_diff = ((f_obs - f_exp)**2 / f_exp).sum()
                        x2[property_to_num(property1)][property_to_num(property2)] = f_diff


                # print(x_df)
                simple_matrix = np.zeros(x2.shape)
                for i, row in enumerate(x2):
                    for j, value in enumerate(row):
                        if value >= 33.196:
                            simple_matrix[i][j] = 1
                        if value >= 36.415:
                            simple_matrix[i][j] = 2
                        if value >= 42.980:
                            simple_matrix[i][j] = 3


                plt.figure(figsize=(25, 25))
                # cmap = ListedColormap(["red", "green", "blue"])
                # ax = sns.heatmap(simple_matrix, annot=False, cmap=cmap, cbar=False, #fmt=".3f",
                #             square=True, xticklabels=FLARE_PROPERTIES, yticklabels=FLARE_PROPERTIES)
                # colorbar = ax.collections[0].colorbar
                # colorbar.set_ticks([0, 1, 2])
                # colorbar.set_ticklabels(['Accept', 'Reject (95% Confidence)', 'Reject (95% Confidence)'])

                colors = {"gray": 0, "red": 1, "yellow": 2, "blue": 3}
                l_colors = sorted(colors, key=colors.get)
                import matplotlib.colors as c
                cMap = c.ListedColormap(l_colors)
                ax = sns.heatmap(simple_matrix, cmap=l_colors, vmin=0, vmax=len(colors), linecolor="black", linewidth=2,
                                 annot=x2, fmt=".4g",
                            square=True, xticklabels=FLARE_PROPERTIES, yticklabels=FLARE_PROPERTIES)
                colorbar = ax.collections[0].colorbar
                colorbar.set_ticks([0, 1, 2, 3])
                colorbar.set_ticklabels(['Accept', 'Reject (90% Confidence)',
                                         'Reject (95% Confidence)',
                                         'Reject (99% Confidence)'])


                plt.title(f"Chi-Square Test of Parameters\n over 24h Average Timeline ({label} Class {is_coincident.capitalize()} Flares)")
                # plt.title(f"{flare_class} Flares (Mean {time_range}h Time Series)")
                plt.tight_layout()
                plt.savefig(f"chi_square/{is_coincident.lower()}/noncorrelative_{label.lower()}_24h_average_chi_square")
                plt.show()
                print(x2)

        generate_binned_csv()
        generate_chi_square()

        # for property1 in FLARE_PROPERTIES:
        #     for property2 in list(set(FLARE_PROPERTIES) - {property1}):
        #         property1_df = data_points[property1]
        #         property2_df = data_points[property2]
        #         print(property1_df)
        #         print(property2_df)
        # print(data_points)




if __name__ == "__main__":
    main()