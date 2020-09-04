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

from scipy.stats import pearsonr
from mobility_model import *
from death_model import *
from defs import *


NORMALIZED = True
ONLY_OECD = 1 # False for oecd
ONLY_US = 0
DEATH = 1 #False for infection
UNTIL_MAY = True

table_ev = 4 # 1 or 2 or 3 or 0 for relying on the settings above

if table_ev == 1:
    ONLY_OECD = 1
    DEATH = 1
    UNTIL_MAY = 1
elif table_ev == 2:
    ONLY_OECD = 1
    DEATH = 0
    UNTIL_MAY = 1
elif table_ev == 3:
    ONLY_OECD = 1
    DEATH = 1
    UNTIL_MAY = False
elif table_ev == 4:
    ONLY_OECD = 1
    DEATH = 0
    UNTIL_MAY = False



def compute_correlations(df, outcome_colname, feature_colnames):
    df_features = pd.DataFrame()
    for feature in feature_colnames:
        r, pv = pearsonr(df[outcome_colname], df[feature])
        df_features.loc[feature, "r"] = r
        df_features.loc[feature, "pv"] = pv
    df_features["r2"] = df_features["r"]**2
    return df_features


if __name__ == '__main__':

    df_mobility, fitted_params_df, fitted_mobility_model_df, mobility_features_df, mobility_timepoints = fit_mobility_model_to_countries(
        transportation_type=TRANSPORTATION_TYPE, US_states=ONLY_US, only_OECD=ONLY_OECD, till_may=True)
    df_death, fitted_death_model_df, death_summary_df, death_timepoints = fit_death_model_to_countries(
        normalized=NORMALIZED, US_states=ONLY_US, countries=df_mobility.index.tolist(), infection=not DEATH, till_may=UNTIL_MAY)

    add_relative_features(mobility_features_df, death_summary_df)
    df = pd.merge(death_summary_df, mobility_features_df, left_index=True,
                  right_index=True)
    bad_df = df.index.isin(['Japan'])
    df_woJapan = df[~bad_df]

    stats_wJapan = (compute_correlations(df, LOG_L_d_POP_COLUMN, list(mobility_features_df.columns) + [RELATIVE_SOCIAL_DISTANCING_START_FEATURE]))
    stats_woJapan = (compute_correlations(df_woJapan, LOG_L_d_POP_COLUMN, list(mobility_features_df.columns) + [RELATIVE_SOCIAL_DISTANCING_START_FEATURE]))

    final = pd.merge(stats_wJapan[["r2", "pv"]], stats_woJapan[["r2", "pv"]], left_index=True, right_index=True, suffixes=("", "woJapan"))
    final.to_csv("corr_matrices_{}.csv".format(int(table_ev)))