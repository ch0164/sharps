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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import plotly.express as px
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

FLARE_LABELS = ["B", "C", "M", "X"]
FLARE_COLORS = ["blue", "green", "orange", "red"]
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


time_range = 12
abc_properties_df = pd.read_csv("../Data_ABC_with_Korsos_parms.csv")
mx_properties_df = pd.read_csv("../Data_MX_with_Korsos_parms.csv")

flare_counts = {label: 0 for label in FLARE_LABELS}


def main():
    # Convert T_REC string to datetime objects.
    abc_properties_df["T_REC"] = \
        abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = \
        mx_properties_df["T_REC"].apply(parse_tai_string)

    info_df = pd.read_csv(f"../all_flare_info_2.csv")
    for time_string in ["time_start", "time_peak", "time_end"]:
        info_df[time_string] = \
            info_df[time_string].apply(parse_tai_string)

    for is_coincident in COINCIDENCES:
        df_needed = pd.DataFrame(columns=FLARE_PROPERTIES)

        temp_df = info_df
        if is_coincident == "coincident":
            info_df = info_df.loc[info_df["is_coincident"] == True]  # ==
        elif is_coincident == "noncoincident":
            info_df = info_df.loc[info_df["is_coincident"] == False]

        for flare_index, row in info_df.iterrows():
            print(flare_index, "/", info_df.shape[0])
            # Find NOAA AR number and timestamp from user input in info dataframe.
            noaa_ar = row["nar"]
            timestamp = floor_minute(row["time_start"]) - datetime.timedelta(0, 12 * 3600)
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
            start_index = end_index
            for i in range(time_range * 5 - 1):
                if end_index - i >= 0:
                    if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
                        start_index = end_index - i

            # Make sub-dataframe of this flare
            # flare = properties_df.iloc[end_index]
            local_properties_df = properties_df.iloc[start_index:end_index + 1]
            new_df = pd.DataFrame(columns=FLARE_PROPERTIES)
            for flare_property in FLARE_PROPERTIES:
                new_df[flare_property] = [local_properties_df[flare_property].mean()]
            new_df['xray_class'] = [flare_class]
            new_df['is_coincident'] = [coincidence]
            df_needed = pd.concat([df_needed, new_df])

        # info_df = temp_df
        df_needed.reset_index(inplace=True)
        df_needed.drop(["index"], axis=1, inplace=True)
        coin_df = df_needed.loc[df_needed["is_coincident"] == True]
        noncoin_df = df_needed.loc[df_needed["is_coincident"] == False]
        df_needed.to_csv(f"singh_prime_flare_data_10-22h_mean_{is_coincident}.csv")
        coin_df.to_csv(f"singh_prime_flare_data_10-22h_mean_{'coincident'}.csv")
        noncoin_df.to_csv(f"singh_prime_flare_data_10-22h_mean_{'noncoincident'}.csv")


def generate_data(coincidence):
    df = pd.read_csv(f"../all_flare_info_2.csv")
    df.dropna(inplace=True)
    print(df)
    hit, miss = 0, 0
    for time_string in ["time_start", "time_peak", "time_end"]:
        df[time_string] = \
            df[time_string].apply(parse_tai_string)

    df_needed = pd.DataFrame()
    for flare_index, row in df.iterrows():
        print(f"{flare_index}/{df.shape[0]}")
        noaa_ar = row["nar"]
        timestamp = floor_minute(row["time_start"]) - datetime.timedelta(0, 2 * 3600)
        flare_class = row["xray_class"]
        is_coincident = row["is_coincident"]

        if flare_class in ["B", "C"]:
            properties_df = abc_properties_df
        else:
            properties_df = mx_properties_df

        # Find corresponding ending index in properties dataframe.
        end_series = properties_df.loc[properties_df["T_REC"] == timestamp]
        if end_series.empty or (
                end_series.loc[end_series['NOAA_AR'] == noaa_ar]).empty:
            miss += 1
            continue
        else:
            hit += 1
            end_index = \
                end_series.loc[end_series['NOAA_AR'] == noaa_ar].index.tolist()[0]

        start_index = end_index
        for i in range(time_range * 5 - 1):
            if end_index - i >= 0:
                if properties_df["NOAA_AR"][end_index - i] == noaa_ar:
                    start_index = end_index - i

        # Make sub-dataframe of this flare
        local_properties_df = properties_df.iloc[start_index:end_index + 1]
        new_df = pd.DataFrame(columns=FLARE_PROPERTIES + ["xray_class", "is_coincident"])
        for flare_property in FLARE_PROPERTIES:
            new_df[flare_property] = [local_properties_df[flare_property].mean()]
        new_df["xray_class"] = flare_class
        new_df["is_coincident"] = is_coincident
        df_needed = pd.concat([new_df, df_needed])
    df_needed.reset_index(inplace=True)
    df_needed.drop("index", axis=1, inplace=True)
    df_needed.to_csv(f"singh_prime_flare_data_10-22h_mean.csv")
    print(hit, miss)


def plot_reduced_scatter_matrix(df, coincidence, experiment):
    n_components = 4
    target = df["xray_class"]
    df = df.drop("xray_class", axis=1)

    pca = PCA(n_components=n_components)
    for name, values in df.iteritems():
        min_value, max_value = df[name].min(), df[name].max()
        df[name] = (df[name] - min_value) / (max_value - min_value)
    components = pca.fit_transform(df)

    total_var = pca.explained_variance_ratio_.sum() * 100
    print(pca.explained_variance_ratio_)

    labels = {str(i): f"PC {i + 1} ({var:.1f})" for i, var in
              zip(range(n_components), pca.explained_variance_ratio_ * 100)}
    labels['color'] = 'X-Ray Class'

    fig = px.scatter_matrix(
        components,
        color=target,
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}% ({coincidence.capitalize()}, {experiment.replace("_", " ")} Before Flare)',
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html(f"pca2/{coincidence}/{experiment}/reduced_scatter_matrix.html")


def plot_scatter_3d(df, coincidence):
    import plotly.graph_objects as go
    target = df["xray_class"]
    df = df.drop(["xray_class", "is_coincident"], axis=1)

    for name, values in df.iteritems():
        min_value, max_value = df[name].min(), df[name].max()
        df[name] = (df[name] - min_value) / (max_value - min_value)
    lda = LinearDiscriminantAnalysis()
    components = lda.fit_transform(df, target)

    ld_labels = [f"LD{i + 1}" for i in range(3)]

    lda_df = pd.DataFrame(components, columns=ld_labels)
    lda_df["xray_class"] = target

    total_var = lda.explained_variance_ratio_.sum() * 100
    # print(len(lda.components_))
    print(lda.explained_variance_ratio_)

    fig = px.scatter_3d(
        lda_df, x="LD1", y="LD2", z="LD3", color=target, opacity=0.5,
        title=f'{coincidence.capitalize()}, 10-22h Mean',
        color_discrete_map={label: color for label, color in zip(FLARE_LABELS, FLARE_COLORS)},
    )

    # Draw sphere
    def ms(x, y, z, radius, resolution=20):
        """Return the coordinates for plotting a sphere centered at (x,y,z)"""
        u, v = np.mgrid[0:2 * np.pi:resolution * 2j, 0:np.pi:resolution * 1j]
        X = radius * np.cos(u) * np.sin(v) + x
        Y = radius * np.sin(u) * np.sin(v) + y
        Z = radius * np.cos(v) + z
        return X, Y, Z

    b_data = lda_df.loc[lda_df["xray_class"] == "B"]
    b_data_count = b_data.shape[0]
    # trim_count = int(0.05 * b_data_count)
    # b_data = b_data[trim_count:-trim_count]

    trim_count = int(0.05 * b_data.shape[0])
    X = sorted(b_data["LD1"])[trim_count:-trim_count]
    Y = sorted(b_data["LD2"])[trim_count:-trim_count]
    Z = sorted(b_data["LD3"])[trim_count:-trim_count]
    b_data["euclid"] = (b_data["LD1"]**2 + b_data["LD2"]**2 + b_data["LD3"]**2)**(1/2)
    x, y, z = np.mean(X) / len(X), np.mean(Y) / len(Y), np.mean(Z) / len(Z)
    r = np.std(X) * 2
    # x -= r

    x_pns_surface, y_pns_surface, z_pns_surface = ms(x, y, z, r)
    fig2 = go.Figure(go.Surface(x=x_pns_surface, y=y_pns_surface, z=z_pns_surface, opacity=0.3, showscale=False))

    fig3 = go.Figure(data=fig.data + fig2.data)
    fig3.update_layout(title_text=f"LDA 10-22h Mean B-Class Flare Spherical Classifier, {coincidence.capitalize()} Flares", title_x=0.5)

    fig3.write_html(f"{coincidence}/lda_3d_modified_trimmed.html")



def spherical_classifier(df, coincidence):
    target = df["xray_class"]
    df = df.drop(["xray_class", "is_coincident"], axis=1)

    for name, values in df.iteritems():
        min_value, max_value = df[name].min(), df[name].max()
        df[name] = (df[name] - min_value) / (max_value - min_value)
    lda = LinearDiscriminantAnalysis()
    components = lda.fit_transform(df, target)
    ld_labels = [f"LD{i + 1}" for i in range(3)]
    lda_df = pd.DataFrame(components, columns=ld_labels)
    lda_df["xray_class"] = target

    b_data = lda_df.loc[lda_df["xray_class"] == "B"]
    b_data_count = b_data.shape[0]
    trim_count = int(0.05 * b_data_count)
    b_data = b_data[trim_count:-trim_count]
    X = sorted(b_data["LD1"])
    Y = sorted(b_data["LD2"])
    Z = sorted(b_data["LD3"])
    cx, cy, cz = np.mean(X) / len(X), np.mean(Y) / len(Y), np.mean(Z) / len(Z)
    r = np.std(X) * 2
    # cx -= r

    threshold = r**2
    inside, outside = [], []
    for index, row in lda_df.iterrows():
        x, y, z, _ = tuple(row)
        result = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        if result <= threshold:
            inside.append(index)
        else:
            outside.append(index)

    tp, fn, fp, tn = 0, 0, 0, 0
    false_positives = {label: 0 for label in ["C", "M", "X"]}
    for index in inside:
        row = lda_df.iloc[index]
        flare_class = row["xray_class"]
        if flare_class == "B":
            tp += 1
        else:
            fp += 1
            false_positives[flare_class] += 1

    for index in outside:
        row = lda_df.iloc[index]
        flare_class = row["xray_class"]
        if flare_class == "B":
            fn += 1
        else:
            tn += 1

    b = len(lda_df.loc[lda_df["xray_class"] == "B"])
    c = len(lda_df.loc[lda_df["xray_class"] == "C"])
    m = len(lda_df.loc[lda_df["xray_class"] == "M"])
    x = len(lda_df.loc[lda_df["xray_class"] == "X"])

    with open(f"{coincidence}/spherical_classification_results_modified_trimmed.txt", "w", newline="\n") as f:
        f.write("Flare Counts\n")
        f.write('-' * 50 + "\n")
        for label, count in zip(FLARE_LABELS, [b, c, m, x]):
            f.write(f"{label}: {count}\n")
        f.write("\n")

        f.write("Binary Classification (B vs. Not-B Flares)\n")
        f.write('-' * 50 + "\n")
        f.write(f'True Positives: {tp}\n')
        f.write(f'True Negatives: {tn}\n')
        f.write(f'False Positives: {fp}\n')
        f.write(f'False Negatives: {fn}\n')
        f.write("\n")

        f.write("False Positives Breakdown\n")
        f.write('-' * 50 + "\n")
        f.write(f'C: {false_positives["C"]}\n')
        f.write(f'M: {false_positives["M"]}\n')
        f.write(f'X: {false_positives["X"]}\n')
        f.write("\n")

        # calculate accuracy
        conf_accuracy = (float(tp + tn) / float(tp + tn + fp + fn))

        # calculate the sensitivity
        conf_sensitivity = (tp / float(tp + fn))
        # calculate the specificity
        conf_specificity = (tn / float(tn + fp))

        conf_false_alarm_rate = (fp / float(fp + tn))

        # calculate precision
        conf_precision = (tn / float(tn + fp))
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        conf_tss = conf_sensitivity - conf_false_alarm_rate

        f.write("Classification Metrics\n")
        f.write('-' * 50 + "\n")
        f.write(f'Accuracy: {round(conf_accuracy, 2)}\n')
        f.write(f'B Recall/Sensitivity: {round(conf_sensitivity, 2)}\n')
        f.write(f'Not-B Recall/Specificity: {round(conf_specificity, 2)}\n')
        f.write(f'Precision: {round(conf_precision, 2)}\n')
        f.write(f'False alarm ratio: {round(conf_false_alarm_rate, 2)}\n')
        f.write(f'f_1 Score: {round(conf_f1, 2)}\n')
        f.write(f"TSS: {round(conf_tss, 2)}\n")

    tp, fn, fp, tn = 0, 0, 0, 0
    false_positives = {label: 0 for label in ["X"]}
    for index in inside:
        row = lda_df.iloc[index]
        flare_class = row["xray_class"]
        if flare_class == "B":
            tp += 1
        elif flare_class == "X":
            fp += 1
            false_positives[flare_class] += 1

    for index in outside:
        row = lda_df.iloc[index]
        flare_class = row["xray_class"]
        if flare_class == "B":
            fn += 1
        elif flare_class == "X":
            tn += 1

    with open(f"{coincidence}/x_spherical_classification_results_modified_trimmed.txt", "w", newline="\n") as f:
        f.write("Flare Counts\n")
        f.write('-' * 50 + "\n")
        for label, count in zip(["B", "X"], [b, x]):
            f.write(f"{label}: {count}\n")
        f.write("\n")

        f.write("Binary Classification (B vs. X Flares)\n")
        f.write('-' * 50 + "\n")
        f.write(f'True Positives: {tp}\n')
        f.write(f'True Negatives: {tn}\n')
        f.write(f'False Positives: {fp}\n')
        f.write(f'False Negatives: {fn}\n')
        f.write("\n")

        f.write("False Positives Breakdown\n")
        f.write('-' * 50 + "\n")
        f.write(f'X: {false_positives["X"]}\n')
        f.write("\n")

        # calculate accuracy
        conf_accuracy = (float(tp + tn) / float(tp + tn + fp + fn))

        # calculate the sensitivity
        conf_sensitivity = (tp / float(tp + fn))
        # calculate the specificity
        conf_specificity = (tn / float(tn + fp))

        conf_false_alarm_rate = (fp / float(fp + tn))

        # calculate precision
        conf_precision = (tn / float(tn + fp))
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        conf_tss = conf_sensitivity - conf_false_alarm_rate

        f.write("Classification Metrics\n")
        f.write('-' * 50 + "\n")
        f.write(f'Accuracy: {round(conf_accuracy, 2)}\n')
        f.write(f'B Recall/Sensitivity: {round(conf_sensitivity, 2)}\n')
        f.write(f'X Recall/Specificity: {round(conf_specificity, 2)}\n')
        f.write(f'Precision: {round(conf_precision, 2)}\n')
        f.write(f'False alarm ratio: {round(conf_false_alarm_rate, 2)}\n')
        f.write(f'f_1 Score: {round(conf_f1, 2)}\n')
        f.write(f"TSS: {round(conf_tss, 2)}\n")

    tp, fn, fp, tn = 0, 0, 0, 0
    false_positives = {label: 0 for label in ["M", "X"]}
    for index in inside:
        row = lda_df.iloc[index]
        flare_class = row["xray_class"]
        if flare_class == "B":
            tp += 1
        elif flare_class == "M" or flare_class == "X":
            fp += 1
            false_positives[flare_class] += 1

    for index in outside:
        row = lda_df.iloc[index]
        flare_class = row["xray_class"]
        if flare_class == "B":
            fn += 1
        elif flare_class == "M" or flare_class == "X":
            tn += 1

    with open(f"{coincidence}/mx_spherical_classification_results_modified_trimmed.txt", "w", newline="\n") as f:
        f.write("Flare Counts\n")
        f.write('-' * 50 + "\n")
        for label, count in zip(FLARE_LABELS, [b, c, m, x]):
            f.write(f"{label}: {count}\n")
        f.write("\n")

        f.write("Binary Classification (B vs. MX Flares)\n")
        f.write('-' * 50 + "\n")
        f.write(f'True Positives: {tp}\n')
        f.write(f'True Negatives: {tn}\n')
        f.write(f'False Positives: {fp}\n')
        f.write(f'False Negatives: {fn}\n')
        f.write("\n")

        f.write("False Positives Breakdown\n")
        f.write('-' * 50 + "\n")
        f.write(f'M: {false_positives["M"]}\n')
        f.write(f'X: {false_positives["X"]}\n')
        f.write("\n")

        # calculate accuracy
        conf_accuracy = (float(tp + tn) / float(tp + tn + fp + fn))

        # calculate the sensitivity
        conf_sensitivity = (tp / float(tp + fn))
        # calculate the specificity
        conf_specificity = (tn / float(tn + fp))

        conf_false_alarm_rate = (fp / float(fp + tn))

        # calculate precision
        conf_precision = (tn / float(tn + fp))
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        conf_tss = conf_sensitivity - conf_false_alarm_rate

        f.write("Classification Metrics\n")
        f.write('-' * 50 + "\n")
        f.write(f'Accuracy: {round(conf_accuracy, 2)}\n')
        f.write(f'B Recall/Sensitivity: {round(conf_sensitivity, 2)}\n')
        f.write(f'MX Recall/Specificity: {round(conf_specificity, 2)}\n')
        f.write(f'Precision: {round(conf_precision, 2)}\n')
        f.write(f'False alarm ratio: {round(conf_false_alarm_rate, 2)}\n')
        f.write(f'f_1 Score: {round(conf_f1, 2)}\n')
        f.write(f"TSS: {round(conf_tss, 2)}\n")



if __name__ == "__main__":
    abc_properties_df["T_REC"] = abc_properties_df["T_REC"].apply(parse_tai_string)
    mx_properties_df["T_REC"] = mx_properties_df["T_REC"].apply(parse_tai_string)
    for coincidence in COINCIDENCES:
        df = pd.read_csv("singh_prime_flare_data_10-22h_mean.csv")
        if coincidence == "coincident":
            df = df.loc[df["is_coincident"] == True]
            df.reset_index(inplace=True)
            df.drop("index", axis=1, inplace=True)
        elif coincidence == "noncoincident":
            df = df.loc[df["is_coincident"] == False]
            df.reset_index(inplace=True)
            df.drop("index", axis=1, inplace=True)
        plot_scatter_3d(df, coincidence)
        spherical_classifier(df, coincidence)
    # main()
