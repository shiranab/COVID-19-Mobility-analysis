#########################################################################
#########################################################################
##                                                                     ##
##                                                                     ##
##   COVID-19 pandemic related lockdown:                               ##
##           response time is more important than its strictness       ##
##                                                                     ##
##                        Analysis code                                ##
##                                                                     ##
## Authors: Gil Loewenthal, Shiran Abadi, Oren Avram,                  ##
##          Keren Halabi, Noa Ecker, Natan Nagar,                      ##
##          Itay Mayrose, and Tal Pupko                                ##
##                                                                     ##
#########################################################################
#########################################################################
## Non-commercial use!                                                 ##
##                                                                     ##
## Please do not change and distribute.                                ##
##                                                                     ##
#########################################################################
#########################################################################
##                                                                     ##
##    Dependencies:                                                    ##
##       python 3.8                                                    ##
##       numpy, pandas, and scipy modules                              ##
##                                                                     ##
#########################################################################


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

TRANSPORTATION_TYPE = "driving"
MOBILITY_DATA_FILE = "data/applemobilitytrends-2020-05-10.csv"  # raw apple data
POP_DATA_FILE = 'data/population_size.csv'
DEATH_DATA_FILE = "data/time_series_covid_19_deaths.csv"  # raw death data
OECD_DATA_FILE = 'data/OECD_countries.csv'
day_dif = 9  # mobility data begins on Jan 13 and death data on Feb 22
thres_death = 10  # number of documented deaths to filter noise

def preprocess_countries_in_death_df(df_death):
    df_death.loc[df_death["Country/Region"] == "Czechia", "Country/Region"] = "Czech Republic"  # rename Czech
    df_death.loc[df_death["Country/Region"] == "Korea, South", "Country/Region"] = "South Korea"  # rename South Korea
    australia_deaths = df_death[df_death["Country/Region"] == "Australia"].sum(axis=0)
    australia_deaths["Country/Region"] = "Australia"
    canada_deaths = df_death[df_death["Country/Region"] == "Canada"].sum(axis=0)
    canada_deaths["Country/Region"] = "Canada"
    us_deaths = df_death[df_death["Country/Region"] == "US"].sum(axis=0)
    us_deaths["Country/Region"] = "United States"
    df_death = df_death[
        (df_death["Country/Region"] != "Australia") & (df_death["Country/Region"] != "Canada") & df_death[
            "Country/Region"] != "US"]
    df_death = df_death[df_death["Province/State"].isna()]
    df_death.reset_index(inplace=True, drop=True)
    df_death.loc[len(df_death)] = australia_deaths
    df_death.loc[len(df_death)] = canada_deaths
    df_death.loc[len(df_death)] = us_deaths
    return df_death


def mobility_model(x, L, t0, k, b, t1, a):
    """
    :param x: data
    :param L: sigmoid drop (on y-axis)
    :param t0: the middle point of decline
    :param k: sigmoid tuning parameter
    :param t1: release day: end time point of sigmoid/ start of back to routine (point on x-axis)
    :param a: linear function: the slope of incline following sigmoid drop
    :param b: sigmoid+linear function: function shift along y axis
    :return:
    """
    y = L / (1 + np.exp(-k * (x - t0))) + b
    if (int(t1) + 1) < len(x) and (t1 > (t0 - np.log(19) / k)):
        y2 = a * x - a * t1 + y[int(t1)]
        y[(int(t1) + 1):] = y2[(int(t1) + 1):]
    return y


def fit_to_mobility_model(xdata, ydata):
    # set starting points (options are 1 or multiple - 3)
    L_0 = max(ydata) - min(ydata)
    x0_0 = np.argwhere(ydata < (100 + min(ydata))/2)[0][0] #Time when M(t)≅ average between 100% and min(M(t))
    k_0 = -1
    b_0 = min(ydata)
    x1_0 = len(xdata) - 30
    a_0 = 1
    starting_points = [L_0, x0_0, k_0, b_0, x1_0, a_0]

    # fit the function
    optimal_solution, _ = curve_fit(mobility_model, xdata, ydata, starting_points, method="lm", maxfev=100000)

    L_opt = optimal_solution[0]
    x0_opt = optimal_solution[1]
    k_opt = optimal_solution[2]
    b_opt = optimal_solution[3]
    x1_opt = optimal_solution[4]
    a_opt = optimal_solution[5]

    return L_opt, x0_opt, k_opt, b_opt, x1_opt, a_opt


def fit_to_death_sig(xdata, ydata):
    # set starting points
    L_0 = max(ydata)
    t0_0 = np.argwhere(ydata >= L_0 / 2)[0][0]
    k_0 = 1
    starting_points = [L_0, t0_0, k_0]

    # fit the function
    def death_sigmoid(x, L, x0, k):
        y = L / (1 + np.exp(-k * (x - x0)))
        return y
    optimal_solution, _ = curve_fit(death_sigmoid, xdata, ydata, starting_points, method="lm", maxfev=100000)

    L_opt = optimal_solution[0]
    x0_opt = optimal_solution[1]
    k_opt = optimal_solution[2]

    return L_opt, x0_opt, k_opt


def extract_mobility_features(xdata, L, t0, k, b, t1, a):
    fitted_values = mobility_model(xdata, L, t0, k, b, t1, a)
    features = {"Social distancing start time": t0 + np.log(19) / k,
                 "Minimal mobility time point": t0 - np.log(19) / k,
                 "Drop duration": abs(np.log(19) / k),
                 "Lockdown strictness": L*100/(L+fitted_values[int(t1)]),
                 "Lockdown duration": t1 - (t0 - np.log(19) / k) if t1 - (t0 - np.log(19) / k) > 0 else 0 ,
                 "Lockdown release day": t1,
                 "Lockdown release rate": a}

    return features


def compute_correlations(df, outcome_colname, feature_colnames):
    df_features = pd.DataFrame()
    for feature in feature_colnames:
        r, pv = pearsonr(df[outcome_colname], df[feature])
        df_features.loc[feature, "r"] = r
        df_features.loc[feature, "pv"] = pv
    df_features["r2"] = df_features["r"]**2
    return df_features


if __name__ == '__main__':
    ##############################  PREPROCESSING  ##############################
    ##### Mobility data
    df_mobility = pd.read_csv(MOBILITY_DATA_FILE)
    df_mobility = df_mobility[(df_mobility.transportation_type == TRANSPORTATION_TYPE)]
    mobility_date_cols = [x for x in df_mobility.columns if "2020" in x]
    mobility_timepoints = np.arange(len(mobility_date_cols))

    df_mobility.loc[df_mobility["region"] == "UK", "region"] = "United Kingdom"
    df_mobility.loc[df_mobility["region"] == "Republic of Korea", "region"] = "South Korea"

    ##### Population data
    df_popsize = pd.read_csv(POP_DATA_FILE, index_col=None, usecols=["Country", "Population"],
                             encoding="ISO-8859-1")
    df_popsize.loc[df_popsize["Country"] == "UK", "Country"] = "United Kingdom"

    ##### Death data
    df_death = pd.read_csv(DEATH_DATA_FILE)
    death_date_cols = [x for x in df_death.columns if '/20' in x]
    death_timepoints = np.arange(len(death_date_cols))
    df_death = preprocess_countries_in_death_df(df_death)

    ##### OECD countries
    df_oecd = pd.read_csv(OECD_DATA_FILE, index_col=None)
    df_oecd = pd.merge(df_oecd, df_popsize, how="left", left_on="name", right_on="Country")
    for col in ["accessionYear", "pop2020", "name"]:
        if col in df_oecd.columns:
            df_oecd.drop(columns=col, inplace=True)

    ############################## FIT MODELS & COMPUTE FEATURES  ##############################
    mobility_features_dict = {}
    for idx, row in df_oecd.iterrows():
        country = row["Country"]
        print(country)

        ### Mobility
        country_mobility = df_mobility.loc[df_mobility.region == country, mobility_date_cols].values.flatten()
        L, t0, k, b, t1, a = fit_to_mobility_model(mobility_timepoints, country_mobility)
        mobility_features_dict[country] = extract_mobility_features(mobility_timepoints, L, t0, k, b, t1, a)

        ### Death
        death_data = df_death.loc[df_death["Country/Region"] == country, death_date_cols].values.flatten()
        L_d, t0_d, k_d = fit_to_death_sig(death_timepoints, death_data)
        df_oecd.loc[idx, "L_d"] = L_d
        df_oecd.loc[idx, "ten_deaths"] = np.argwhere(death_data > (thres_death - 1))[0] + day_dif

    mobility_features_df = pd.DataFrame(mobility_features_dict).T
    feature_names = mobility_features_df.columns

    df_oecd["L_d/pop"] = df_oecd["L_d"]/df_oecd["Population"]
    df_oecd["L_d/pop__log"] = df_oecd["L_d/pop"].apply(np.log10)

    df = pd.merge(df_oecd, mobility_features_df, left_on="Country", right_index=True, sort=True)

    df["Relative social distancing start time (tau)"] = df["Social distancing start time"] - df["ten_deaths"]
    df["Relative minimal mobility time point"] = df["Minimal mobility time point"] - df["ten_deaths"]
    df["Relative lockdown release day"] = df["Lockdown release day"] - df["ten_deaths"]

    feature_names = list(feature_names) + ["Relative social distancing start time (tau)", "Relative minimal mobility time point", "Relative lockdown release day"]

    df_features = compute_correlations(df, "L_d/pop__log", feature_names)
    df_woJapan = df[df["Country"] != 'Japan']
    df_features_woJapan = compute_correlations(df_woJapan, "L_d/pop__log", feature_names)
    final = pd.merge(df_features, df_features_woJapan, left_index=True, right_index=True, suffixes=("", " - without Japan"))

    final.to_csv("social distancing X death - regression analysis.csv")

    # Regression analysis
    b, m = np.polyfit(df_woJapan["Relative social distancing start time (tau)"], df_woJapan["L_d/pop__log"], 1)
    print("y={:.2f}x+{:.2f}".format(m, b))
