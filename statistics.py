import pandas as pd
from scipy.stats import differential_entropy

flare_labels = ["B", "C", "M", "X"]

coincidences = ["all", "coincident", "noncoincident"]

idealized_flares = "idealized_flares2"

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

to_drop = ['d_l_f', 'g_s', 'slf']

mean_df = pd.DataFrame(columns=FLARE_PROPERTIES)
median_df = pd.DataFrame(columns=FLARE_PROPERTIES)
std_df = pd.DataFrame(columns=FLARE_PROPERTIES)
delta6_df = pd.DataFrame(columns=FLARE_PROPERTIES)
delta12_df = pd.DataFrame(columns=FLARE_PROPERTIES)
delta24_df = pd.DataFrame(columns=FLARE_PROPERTIES)
entropy_df = pd.DataFrame(columns=FLARE_PROPERTIES)

statistics = {
    "mean": mean_df,
    "median": median_df,
    "std": std_df,
    "delta24h": delta24_df,
    "delta12h": delta12_df,
    "delta6h": delta6_df,
    "entropy": entropy_df,
}


def generate_idealized_statistics(is_coincident):
    for stat_label in statistics.keys():
        for flare_label in flare_labels:
            flare_df = pd.read_csv(f"{idealized_flares}/{is_coincident}/{flare_label}_idealized_flare.csv")
            flare_df.drop(to_drop, inplace=True, axis=1)
            flare_df.drop("Unnamed: 0", inplace=True, axis=1)
            flare_df.drop([0], inplace=True, axis=0)

            df = pd.DataFrame(columns=FLARE_PROPERTIES)
            for flare_property in FLARE_PROPERTIES:
                if stat_label in "mean":
                    df[flare_property] = [flare_df[flare_property].mean()]
                elif stat_label in "median":
                    df[flare_property] = [flare_df[flare_property].median()]
                elif stat_label in "std":
                    df[flare_property] = [flare_df[flare_property].std()]
                elif stat_label in "delta24h":
                    t0 = flare_df[flare_property][1]
                    t1 = flare_df[flare_property][flare_df.shape[0]]
                    delta = t1 - t0
                    df[flare_property] = [delta]
                elif stat_label in "delta12h":
                    t0 = flare_df[flare_property][1]
                    t1 = flare_df[flare_property][flare_df.shape[0] // 2]
                    t2 = flare_df[flare_property][flare_df.shape[0]]
                    delta1 = t1 - t0
                    delta2 = t2 - t1
                    df[flare_property] = [delta1, delta2]
                elif stat_label in "delta6h":
                    t0 = flare_df[flare_property][1]
                    t1 = flare_df[flare_property][flare_df.shape[0] // 4]
                    t2 = flare_df[flare_property][flare_df.shape[0] // 2]
                    t3 = flare_df[flare_property][(flare_df.shape[0] // 2) + (flare_df.shape[0] // 4)]
                    t4 = flare_df[flare_property][flare_df.shape[0]]
                    delta1 = t1 - t0
                    delta2 = t2 - t1
                    delta3 = t3 - t2
                    delta4 = t4 - t3
                    df[flare_property] = [delta1, delta2, delta3, delta4]
                elif stat_label in "entropy":
                    df[flare_property] = [differential_entropy(flare_df[flare_property])]
            stat_df = statistics[stat_label]
            statistics[stat_label] = pd.concat([stat_df, df])

    for stat_label, stat_df in statistics.items():
        if stat_label in "delta12h":
            new_labels = [f"{label}{number}" for label in flare_labels for number in [1, 2]]
        elif stat_label in "delta6h":
            new_labels = [f"{label}{number}" for label in flare_labels for number in [1, 2, 3, 4]]
        else:
            new_labels = flare_labels
        stat_df.index = new_labels
        print(stat_label, stat_df)
        stat_df.to_csv(f"statistics/{is_coincident}/{stat_label}_{is_coincident}.csv")
        statistics[stat_label] = pd.DataFrame(columns=FLARE_PROPERTIES)


def main():
    for is_coincident in coincidences:
        generate_idealized_statistics(is_coincident)


if __name__ == "__main__":
    main()
