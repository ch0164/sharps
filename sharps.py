import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib
from astropy.io import fits
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import sunpy.map
import sunpy.io
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

KEY_VALUES = ["USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", 'MEANGBH', 'MEANJZD', 'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'TOTPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE']

poly_reg = PolynomialFeatures(degree=2)

TIMES = [
    "[2010.05.05_15:00:00/12h]", "[2011.02.14_15:00:00/12h]",
    "[2012.05.06_17:00:00/12h]", "[2013.12.22_14:00:00/12h]",
    "[2015.07.03_12:00:00/12h]", "[2021.10.29_02:00:00/12h]",
]

def plot_model(X: npt.ArrayLike, Y: npt.ArrayLike,
               a: float, b: float, c: str, key: str) -> None:
    """Plot the least squares line with the given color and data points."""
    plt.xlabel("TIME")
    plt.ylabel(key)

    least_squares_line = b * X + a
    plt.scatter(X, Y, color="r", marker="o", s=30)
    plt.plot(X, least_squares_line, color=c)
    plt.savefig(f"{key}_over_time")


def plot_quad_model(X: npt.ArrayLike, Y: npt.ArrayLike,
               a: float, b: float, c: str, key: str) -> None:
    """Plot the least squares line with the given color and data points."""
    plt.xlabel("TIME")
    plt.ylabel(key)

    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, Y)

    least_squares_line = b * X + a
    plt.scatter(X, Y, color="r", marker="o", s=30)
    plt.plot(X, least_squares_line, color=c)
    plt.savefig(f"{key}_over_time")


def parse_tai_string(tstr,datetime=True):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    if datetime: return dt_obj(year,month,day,hour,minute)
    else: return year,month,day,hour,minute


if __name__ == "__main__":
    c = drms.Client()

    s = c.series(r"hmi\.sharp_")[3]

    si = c.info(s)


    keys, segments = c.query('hmi.sharp_cea_720s[377][2011.02.14_15:00:00/12h]',
        key='T_REC, USFLUX, MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZD, TOTUSJZ, MEANALP, MEANJZH, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, MEANSHR, SHRGT45, R_VALUE', 
        seg='Br')

    #print(type(keys["R_VALUE"]), type(keys["TOTUSJH"]))

    X = np.array([parse_tai_string(keys.T_REC[i],datetime=True).toordinal() for i in range(keys.T_REC.size)])
    #X = np.array([int(round(dt.timestamp())) for dt in t_rec])
    for key in KEY_VALUES:
        Y = keys[key]
        reg = LinearRegression().fit(np.reshape(X, (-1, 1)), Y)
        alpha, beta = reg.intercept_, reg.coef_[0]
        # print(f"Sklearn Model Coefficients: alpha = {alpha}, beta = {beta}\n")

        plot_model(X, Y, alpha, beta, "b", key)
        plt.clf()
    #Y = keys["USFLUX"].to_numpy()
    #Y = keys["R_VALUE"].to_numpy()
    #Y = keys["TOTUSJH"].to_numpy()

    

    # fig, ax = plt.subplots(figsize=(8,7))      # define the size of the figure
    # orangered = (1.0,0.27,0,1.0)                # create an orange-red color

    # # define some style elements
    # marker_style = dict(linestyle='', markersize=8, fillstyle='full',color=orangered,markeredgecolor=orangered)
    # text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
    # ax.tick_params(labelsize=14)
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    # # ascribe the data to the axes
    # ax.plot(t_rec, (keys.USFLUX)/(1e22),'o',**marker_style)
    # ax.errorbar(t_rec, (keys.USFLUX)/(1e22), yerr=(keys.ERRVF)/(1e22), linestyle='',color=orangered)

    # # format the x-axis with universal time
    # locator = AutoDateLocator()
    # locator.intervald[HOURLY] = [3] # only show every 3 hours
    # formatter = DateFormatter('%H')
    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)

    # # set yrange 
    # ax.set_ylim([2.4,2.8])

    # # label the axes and the plot
    # ax.set_xlabel('time in UT',**text_style)
    # ax.set_ylabel('maxwells x 1e22',**text_style)
    # ax.set_title('total unsigned flux starting at '+str(t_rec[0])+' UT',**text_style) # annotate the plot with a start time