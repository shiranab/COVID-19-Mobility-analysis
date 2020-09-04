from defs import *


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
    optimal_solution, pcov = curve_fit(mobility_model, xdata, ydata, starting_points, method="lm", maxfev=100000)

    L_opt = optimal_solution[0]
    x0_opt = optimal_solution[1]
    k_opt = optimal_solution[2]
    b_opt = optimal_solution[3]
    x1_opt = optimal_solution[4]
    a_opt = optimal_solution[5]

    return L_opt, x0_opt, k_opt, b_opt, x1_opt, a_opt


# extracts features based on sigmoid plus fit
def extract_mobility_features(L, t0, k, b, t1, a, fitted_values):
    features = {"Social distancing start time": t0 + np.log(19) / k,
                 "Minimal mobility time point": t0 - np.log(19) / k,
                 "Drop duration": abs(np.log(19) / k),
                 "Lockdown strictness": L*100/(L+fitted_values[min(int(t1), len(fitted_values)-1)]),
                 "Lockdown duration": t1 - (t0 - np.log(19) / k) if t1 - (t0 - np.log(19) / k) > 0 else 0 ,
                 "Lockdown release day": t1,
                 "Lockdown release rate": a}

    return features


def read_mobility_dataset(transportation_type=TRANSPORTATION_TYPE, countries=None, US_states=False, only_OECD=True, till_may=True):
    df_mobility = pd.read_csv(MOBILITY_DATA_FILE)
    # Fill in the missing dates May 11-12 according to May 10 and 13
    df_mobility["2020-05-11"] = df_mobility["2020-05-10"] + (df_mobility["2020-05-13"]-df_mobility["2020-05-10"])/3
    df_mobility["2020-05-12"] = df_mobility["2020-05-10"] + 2*(df_mobility["2020-05-13"]-df_mobility["2020-05-10"])/3

    df_mobility = df_mobility[(df_mobility.transportation_type == transportation_type)]
    if not US_states:
        df_mobility = df_mobility[df_mobility["geo_type"] == "country/region"]
    else:
        df_mobility = df_mobility[df_mobility["geo_type"] == "sub-region"]
        countries = US_STATES_LIST

    mobility_date_cols = [x for x in df_mobility.columns if "20" in x]
    if till_may:
        mobility_date_cols = mobility_date_cols[:mobility_date_cols.index("2020-05-10")+1]
    else:
        mobility_date_cols = mobility_date_cols[:mobility_date_cols.index("2020-08-31")+1]
    mobility_timepoints = np.arange(len(mobility_date_cols))

    df_mobility.loc[df_mobility["region"] == "UK", "region"] = "United Kingdom"
    df_mobility.loc[df_mobility["region"] == "Macao", "region"] = "Macau"
    df_mobility.loc[df_mobility["region"] == "Republic of Korea", "region"] = "South Korea"

    if countries:
        df_mobility = df_mobility.loc[df_mobility["region"].isin(countries)]
    elif not US_states and only_OECD:
        df_oecd = pd.read_csv(OECD_DATA_FILE, index_col=None)
        df_mobility = df_mobility.loc[df_mobility["region"].isin(df_oecd["name"])]

    df_mobility.set_index(keys="region", drop=True, inplace=True)
    df_mobility = df_mobility[mobility_date_cols]

    return df_mobility, mobility_timepoints, mobility_date_cols


def fit_mobility_model_to_countries(transportation_type=TRANSPORTATION_TYPE, countries=None, US_states=False, only_OECD=True, till_may=True):
    df_mobility, mobility_timepoints, mobility_date_cols = \
        read_mobility_dataset(transportation_type=transportation_type, countries=countries, US_states=US_states,
                              only_OECD=only_OECD, till_may=till_may)

    fitted_params_df = pd.DataFrame(index=df_mobility.index, columns=["L", "t0", "k", "b", "t1", "a"])
    fitted_model_df = pd.DataFrame(index=df_mobility.index, columns=mobility_date_cols)

    ############################## FIT MODELS & COMPUTE FEATURES  ##############################
    mobility_features_dict = {}
    for country, row in df_mobility.iterrows():
        country_mobility = row.values.flatten()
        L, t0, k, b, t1, a = fit_to_mobility_model(mobility_timepoints, country_mobility)
        fitted_values = mobility_model(mobility_timepoints, L, t0, k, b, t1, a)
        mobility_features_dict[country] = extract_mobility_features(L, t0, k, b, t1, a, fitted_values)

        fitted_params_df.loc[country, ["L", "t0", "k", "b", "t1", "a"]] = L, t0, k, b, t1, a
        fitted_model_df.loc[country, mobility_date_cols] = fitted_values

    mobility_features_df = pd.DataFrame(mobility_features_dict).T

    return df_mobility, fitted_params_df, fitted_model_df, mobility_features_df, mobility_date_cols


def add_relative_features(mobility_features_df, df_death_summary):
    #### Note: only 63 countries have mobility data
    merged_df = pd.merge(df_death_summary, mobility_features_df, left_index=True, right_index=True)
    df_death_10deaths_day = df_death_summary.loc[merged_df.index, TEN_DEATHS_COLUMN]

    mobility_features_df.loc[merged_df.index, RELATIVE_SOCIAL_DISTANCING_START_FEATURE] = mobility_features_df.loc[merged_df.index, SOCIAL_DISTANCING_START_FEATURE] - df_death_10deaths_day
    mobility_features_df.loc[merged_df.index, "Relative minimal mobility time point"] = mobility_features_df.loc[merged_df.index, "Minimal mobility time point"] - df_death_10deaths_day
    mobility_features_df.loc[merged_df.index, "Relative lockdown release day"] = mobility_features_df.loc[merged_df.index, "Lockdown release day"] - df_death_10deaths_day

    return df_death_10deaths_day