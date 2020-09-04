### data sources:
# death/confirmed: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
# mobility: https://covid19.apple.com/mobility
# population size: https://worldpopulationreview.com/

MOBILITY_DATA_FILE = "data/applemobilitytrends-2020-08-31.csv" # raw apple data
DEATH_DATA_FILE = r"data/time_series_covid19_deaths_global.csv" # raw death data
CONFIRMED_DATA_FILE = r"data/time_series_covid19_confirmed_global.csv" # raw death data
US_DEATH_DATA_FILE = r"data/time_series_covid19_deaths_US.csv" # raw death data
US_CONFIRMED_DATA_FILE = r"data/time_series_covid19_confirmed_US.csv" # raw death data
POP_DATA_FILE = 'data/population_size.csv'  #, encoding = "ISO-8859-1") #population data
OECD_DATA_FILE = 'data/OECD_countries.csv'

DEATH_MOBILITY_DAY_DIFF = 9
DEATH_THRES = 10
INFECTION_THRES = 500
TRANSPORTATION_TYPE = 'driving'

# Headers
SOCIAL_DISTANCING_START_FEATURE = "Social distancing start time"
MIN_MOBILITY_TIME_FEATURE = "Minimal mobility time point"
RELATIVE_SOCIAL_DISTANCING_START_FEATURE = "Relative social distancing start time (tau)"
TEN_DEATHS_COLUMN = "ten_deaths"
POPULATION_COLUMN = "Population"
L_d_POP_COLUMN = "L_d/pop"
LOG_L_d_POP_COLUMN = "L_d/pop__log"

colors = ["#ef9f00", "#cc79a7", "#56b4e9", "#009e73", "#d55e00", "#f0e442", "#0072b2"]
death_data_color = "lightsteelblue"
mobility_data_color = "peachpuff"
mobility_color = colors[4]
death_color = "navy"
daily_death_color = colors[-1]


US_STATES_LIST = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
             'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
             'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
             'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
             'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
             'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

from matplotlib.ticker import FixedLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import lmfit
from scipy.stats import pearsonr, spearmanr