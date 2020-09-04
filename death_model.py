from defs import *


def death_sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y


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


def read_popsize(countries=None):
    df_popsize = pd.read_csv(POP_DATA_FILE, index_col=None, usecols=["Country", "Population"],
                             encoding="ISO-8859-1")
    df_popsize.loc[df_popsize["Country"] == "UK", "Country"] = "United Kingdom"
    df_popsize.loc[len(df_popsize)] = ["Cote d'Ivoire", 26348445] #Wikipedia
    df_popsize.loc[len(df_popsize)] = ["Diamond Princess", 3700] #https://watermark.silverchair.com/taaa030.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAArMwggKvBgkqhkiG9w0BBwagggKgMIICnAIBADCCApUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM3vgpcDcTxZuLE9wtAgEQgIICZv3qraqcfOk3ZyPHPzcs8uQrBTajqeNrYXsK9qbrxXWmiAQUd6CHo7Ym8GkuMnL-5HOJhBN2FtyIYSJrRkaXbLFAggVpLBLIxeOP3f0rV_VPb4myqbWmTpDBmZCASHiFxVOtq8g_YY7YDoCUEWpTxu0MQbh5DI51WtDzM97axPOxCuwWG8KH43fPpi3JHPfWqty0cjRruM4wCIigA7ybh7Erhyg5GEHj90IeiF14S44JyeLP9NOggfDQC0J9G6xGktQghZ6Kx9m81Jd58ah1W3txFCZTG-QAETo5UfHkIPvz_XBySBnXcQF4P5wXDVYjcr6zkX_OAJuWN7VVdJ7pAQzdhemmAz8jAH7G_49eMZgQsmvhX3LOMu_4Kf9c8qvtWhTd7RWU9CqhNNE-itcFO6HxFQ7lXn4m11SvDu4EQj8XODXzWr1bN4tLCxpne_9s8dXhOTHsOwMObOaRh55P49N3RuTs70uziouOqGy6wXmiaa-yRqZBN4Vqk7B79XDVOINdvhQbxj3ecINZ6f-COwYZd0oGYp1abk1B2U6frbTlQic61HodX1DcZvSEGeLTepb_4LQovfDj6xlyXADuwrqQD_UJrNu9_OZ64TkbYojXzOHl4STSPwrWv1TwI34n5bGqm24W1p1HkUeQxBHRBSQIZuYgutG1NeOpVxnlmyMmqhELQ93di8cl7rB6-hhsZM4tmAeVhXejyJqQMtElKHO_fHRB9A3SJZhZ_Lr_Gvo20KKxgyO2SR8L4V3BsO1Tr6QxzmHXDnc5lzFHTM0eIz4PXmVuAC7i4INVF47OvaBnQ60Y_coK
    df_popsize.loc[len(df_popsize)] = ["Kosovo", 1810366] #Wikipedia

    if countries:
        df_popsize = df_popsize.loc[df_popsize["Country"].isin(countries)]
    df_popsize.set_index(keys="Country", drop=True, inplace=True)

    return df_popsize[["Population"]]


def read_death_data(countries=None, US_states=False, infection=False, till_may=True):
    if not infection:
        df_death = pd.read_csv(DEATH_DATA_FILE)
        df_us_death = pd.read_csv(US_DEATH_DATA_FILE)
    else:
        df_death = pd.read_csv(CONFIRMED_DATA_FILE)
        df_us_death = pd.read_csv(US_CONFIRMED_DATA_FILE)

    df_us_death.rename(columns={"Country_Region":"Country/Region", "Province_State": "Province/State"}, inplace=True)
    death_date_cols = [x for x in df_death.columns if '/20' in x]
    if not till_may:
        death_date_cols = death_date_cols[:death_date_cols.index("8/31/20")+1]
    else:
        death_date_cols = death_date_cols[:death_date_cols.index("5/10/20")+1]

    death_timepoints = np.arange(len(death_date_cols))
    if not US_states:
        df_death = pd.concat([df_death, df_us_death], axis=0,)
        df_death = preprocess_countries_in_death_df(df_death)
    else:
        df_death = df_us_death
        df_death = preprocess_US_countries_in_death_df(df_death)
        countries = US_STATES_LIST

    if countries:
        df_death = df_death.loc[df_death["Country/Region"].isin(countries)]
    df_death.set_index(keys="Country/Region", drop=True, inplace=True)
    df_death = df_death[death_date_cols]

    return df_death, death_timepoints, death_date_cols


def fit_death_model_to_countries(countries=None, normalized=False, US_states=False, infection=False, till_may=True):
    df_death, death_timepoints, death_date_cols = \
        read_death_data(countries=countries, US_states=US_states, infection=infection, till_may=till_may)

    if US_states:
        countries = US_STATES_LIST
        countries.remove("Wyoming")
        countries.remove("South Dakota")
        df_death = df_death.loc[df_death.index != "Wyoming"]
        df_death = df_death.loc[df_death.index != "South Dakota"]
    if not infection:
        thres = DEATH_THRES
    else:
        thres = INFECTION_THRES

    death_summary_df = read_popsize(countries)
    fitted_model_df = pd.DataFrame(index=df_death.index, columns=death_date_cols)

    succeeded = []
    for country, row in df_death.iterrows():
        death_data = row.values.flatten()
        L_d, t0_d, k_d = fit_to_death_sig(death_timepoints, death_data)
        # death_summary_df.loc[country, "t0_d"] = t0_d
        # death_summary_df.loc[country, "k_d"] = k_d
        death_summary_df.loc[country, "L_d"] = L_d

        if len(np.argwhere(death_data > (thres - 1))) > 0:
            death_summary_df.loc[country, TEN_DEATHS_COLUMN] = np.argwhere(death_data > (thres - 1))[0] + DEATH_MOBILITY_DAY_DIFF
        else:
            print("Attention: country", country, "doesn't have 10 deaths")
            continue

        fitted_model_df.loc[country, death_date_cols] = death_sigmoid(death_timepoints, L_d, t0_d, k_d)

        # r2, pv = pearsonr(death_data, fitted_model_df.loc[country, death_date_cols])
        # death_summary_df.loc[country, "Pearson r2"] = r2
        # death_summary_df.loc[country, "pv"] = pv

        succeeded.append(country)
    df_death = df_death.loc[succeeded]
    death_summary_df = death_summary_df.loc[succeeded]
    fitted_model_df = fitted_model_df.loc[succeeded]

    death_summary_df[TEN_DEATHS_COLUMN] = death_summary_df[TEN_DEATHS_COLUMN].astype(int)

    if not till_may:
        death_summary_df["L_d"] = df_death[death_date_cols].sum(axis=1)
    death_summary_df[L_d_POP_COLUMN] = death_summary_df["L_d"] / death_summary_df[POPULATION_COLUMN]

    if normalized:
        df_death = df_death.divide(death_summary_df.loc[df_death.index, POPULATION_COLUMN], axis=0) * 10**6
        fitted_model_df = fitted_model_df.divide(death_summary_df.loc[fitted_model_df.index, POPULATION_COLUMN], axis=0) * 10**6
        death_summary_df[L_d_POP_COLUMN] *= 10**6
    death_summary_df[LOG_L_d_POP_COLUMN] = death_summary_df[L_d_POP_COLUMN].apply(np.log10)

    return df_death, fitted_model_df, death_summary_df, death_timepoints


def preprocess_countries_in_death_df(df_death):
    lst = []
    df_death.loc[df_death["Country/Region"].str.startswith("Congo"), "Country/Region"] = "Congo"  # Congo sub-regions
    df_death.loc[df_death["Country/Region"].str.startswith("Taiwan"), "Country/Region"] = "Taiwan"  # Congo sub-regions
    df_death.loc[df_death["Province/State"] == "Hong Kong", "Country/Region"] = "Hong Kong"  # Hong Kong appears as part of China
    df_death.loc[df_death["Province/State"] == "Macau", "Country/Region"] = "Macau" # Macau appears as part of China
    for unique_country in df_death["Country/Region"].unique():
        country_deaths = df_death[df_death["Country/Region"] == unique_country].sum(axis=0)
        country_deaths["Country/Region"] = unique_country
        country_deaths["Province/State"] = ""
        lst.append(country_deaths)

    df_death = pd.concat(lst, axis=1, ignore_index=True).T
    df_death.reset_index(inplace=True, drop=True)

    df_death.loc[df_death["Country/Region"] == "Czechia", "Country/Region"] = "Czech Republic"  # rename Czech
    df_death.loc[df_death["Country/Region"] == "Korea, South", "Country/Region"] = "South Korea"  # rename South Korea
    df_death.loc[df_death["Country/Region"] == "US", "Country/Region"] = "United States"  # rename South Korea
    return df_death


def preprocess_US_countries_in_death_df(df_death):
    df_death = df_death[df_death["Country/Region"]== "US"]
    new_death_df = pd.DataFrame(columns=df_death.columns)
    for state in df_death["Province/State"].unique():
        state_deaths = df_death[df_death["Province/State"] == state].sum(axis=0)
        state_deaths["Country/Region"] = state
        state_deaths["Province/State"] = state
        new_death_df.loc[len(new_death_df)] = state_deaths

    return new_death_df


if __name__ == '__main__':
    from mobility_model import fit_mobility_model_to_countries, add_relative_features

    df_oecd = pd.read_csv(OECD_DATA_FILE, index_col=None)
    countries = sorted(df_oecd["name"])

    df_mobility, fitted_params_df, fitted_mobility_model_df, mobility_features_df, mobility_timepoints = fit_mobility_model_to_countries(
        transportation_type=TRANSPORTATION_TYPE, US_states=False, only_OECD=True, till_may=True, countries=countries)

    df_death, fitted_death_model_df, death_summary_df, death_timepoints = fit_death_model_to_countries(
        normalized=False, US_states=False, countries=df_mobility.index.tolist())

    add_relative_features(mobility_features_df, death_summary_df)
    death_summary_df.to_csv("death_summary.csv")
    mobility_features_df.to_csv("mobility_summary.csv")
