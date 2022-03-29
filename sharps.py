import drms
import json, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
from datetime import datetime as dt_obj
import urllib
from astropy.io import fits
from sunpy.visualization.colormaps import color_tables as ct
from matplotlib.dates import *
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import sunpy.map
import sunpy.io
from IPython.display import Image
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# 20 parameters
KEY_VALUES = [
    'ABSNJZH',
    'MEANALP',
    'MEANGAM',
    'MEANGBH',
    'MEANGBT',
    'MEANGBZ',
    'MEANJZD',
    'MEANJZH',
    'MEANPOT',
    'MEANSHR',
    'NACR',
    'R_VALUE',
    'SAVNCPP',
    'SHRGT45',
    'SIZE',
    'SIZE_ACR',
    'TOTPOT',
    'TOTUSJH',
    'TOTUSJZ',
    'USFLUX'
]

# Times obtained using https://www.spaceweatherlive.com/

GEOMAGNETIC_STORM_TIMES = [
    "2012.03.09_06:00:00/12h",
    "2015.06.22_06:00:00/12h",
    "2017.09.08_06:00:00/12h",
]

SOLAR_FLARE_TIMES = [
    "2012.03.06_18:00:00/12h",  # X5.4
    "2014.02.24_18:30:00/12h",  # X4.9
    "2017.09.06_06:00:00/12h",  # X9.3
]

SUN_SPOT_TIMES = [
    "2011.09.25_06:00:00/12h",  # Size=1300, n=12
    "2012.07.13_06:00:00/12h",  # Size=1460, n=53
    "2014.10.27_06:00:00/12h",  # Size=2750, n=60
]

SUN_SPOTLESS_DAYS = [
    "2014.11.15_06:00:00/12h",
    "2014.12.01_06:00:00/12h",
    "2014.12.21_06:00:00/12h",
]

TIMES = [GEOMAGNETIC_STORM_TIMES, SOLAR_FLARE_TIMES, SUN_SPOT_TIMES, SUN_SPOTLESS_DAYS]
LABELS = ["geomagnetic_storm", "solar_flare", "sun_spot", "sun_spotless"]

KEY_VALUES_TREC = KEY_VALUES + ["T_REC"]

def parse_tai_string(tstr,datetime=True):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    if datetime:
        d = dt_obj(year,month,day,hour,minute)
        return d
    else: return year,month,day,hour,minute


if __name__ == "__main__":
    series = 'hmi.sharp_cea_720s'
    sharpnum = ""  # sharp number
    segments = ['magnetogram', 'continuum']
    c = drms.Client()

    for dir, times in zip(LABELS, TIMES):
        for time in times:
            # [? (QUALITY<65536) ?]
            keys = c.query(f'{series}[{sharpnum}][{time}][? (QUALITY<65536) ?]', key=KEY_VALUES_TREC, rec_index=True)
            t_rec = np.array([parse_tai_string(keys.T_REC[i], datetime=True) for i in range(keys.T_REC.size)])
            keys.replace([np.inf, -np.inf], np.nan, inplace=True)
            keys = keys.dropna()
            time_series = np.array(list(range(1, keys.T_REC.size + 1)))
            print(time)

            for key in KEY_VALUES:
                Y = keys[key]
                reg = LinearRegression().fit(np.reshape(time_series, (-1, 1)), Y)
                alpha, beta = reg.intercept_, reg.coef_[0]

                least_squares_line = beta * time_series + alpha
                least_squares_parabola = np.poly1d(np.polyfit(time_series, Y, 2))
                least_squares_cubic = np.poly1d(np.polyfit(time_series, Y, 3))

                plt.plot(time_series, least_squares_line, c="b")
                plt.plot(time_series, least_squares_parabola(time_series), c="g")
                plt.plot(time_series, least_squares_cubic(time_series), c="m")

                plt.legend(["line", "parabola", "cubic"])

                plt.scatter(time_series, Y, c="r")

                plt.title(f"Total {key} From {t_rec[0]} to {t_rec[-1]} UT")
                plt.xlabel("Timepoint")
                plt.ylabel(f"{key}")

                plt.savefig(f"linear_regression/{dir}/{key}_{time[:-4].replace(':', '_').replace('.', '_')}")
                plt.clf()
        # print(f"Sklearn Model Coefficients: alpha = {alpha}, beta = {beta}\n")

    # print("Keys:", type(k))
    #
    # t1 = t_rec[0]
    # t2 = t_rec[-1]
    # plt.plot_date(t_rec, k.USFLUX, xdate=True)
    # #plt.show()
    # #print("Time Difference:", t2 - t1)
    # rec_cm = k.USFLUX.abs().idxmin()
    # k_cm = k.loc[rec_cm]
    # print(k_cm)
    # print(rec_cm, '@', k.USFLUX[rec_cm], 'deg')
    # print('Timestamp:', t1)

    # keys, segments = c.query('hmi.sharp_cea_720s[377][2011.02.14_15:00:00/12h]',
    #     key='T_REC, USFLUX, MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZD, TOTUSJZ, MEANALP, MEANJZH, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, MEANSHR, SHRGT45, R_VALUE',
    #     seg='Br')
    #
    # #print(type(keys["R_VALUE"]), type(keys["TOTUSJH"]))
    #
    # X = np.array([parse_tai_string(keys.T_REC[i],datetime=True) for i in range(keys.T_REC.size)])
    # #X = np.array([int(round(dt.timestamp())) for dt in t_rec])

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




