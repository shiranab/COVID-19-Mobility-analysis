# COVID-19-Mobility-analysis
### Analysis for our manuscript: COVID 19 pandemic related lockdown: response time is more important than its strictness

#### See our paper: https://www.medrxiv.org/content/10.1101/2020.06.11.20128520v1

#### Authors: Gil Loewenthal, Shiran Abadi, Oren Avram, Keren Halabi, Noa Ecker, Natan Nagar, Itay Mayrose, and Tal Pupko

### Abstract
The rapid spread of SARS-CoV-2 and its threat to health systems worldwide have led governments to take acute actions to enforce social distancing. Previous studies used complex epidemiological models to quantify the effect of lockdown policies on infection rates. However, these rely on prior assumptions or on official regulations. Here, we use country-specific reports of daily mobility from people cellular usage to model social distancing. Our data-driven model enabled the extraction of lockdown characteristics which were crossed with observed mortality rates to show that: (1) the time at which social distancing was initiated is highly correlated with the number of deaths, r^2=0.64, while the lockdown strictness or its duration are not as informative; (2) a delay of 7.49 days in initiating social distancing would double the number of deaths; and (3) the immediate response has a prolonged effect on COVID-19 death toll.


The attached code fits the proposed models to the mobility data (Apple) and the daily deaths data (John's Hopkins University), extracts the relevant features, and runs the correlation between each feature and the fit of the regression analysis (paper, figure 4).

### How to run?
Change the parameters (listed below) in file "correlate_features.py" and run it!


### Parameters:
* ONLY_OECD - if True, run the analysis only for OECD countries, otherwise for all countries for which data exist.
* ONLY_US - if True, run the analysis for US states for which data exist.
* NORMALIZE - if True, normalize the death/confirmed data to 1 million population, otherwise, only divide the expected number by the population size.
* DEATH - if True, use #COVID-19 deaths, otherwise use #COVID-19 confirmed cases.
* UNTIL_MAY - if True, fit the logistic death function to the deaths/confirmed daily reports until May 10, 2019. Otherwise, use the raw data until August 31, 2020.


### Data update:
If you wish to replace the data files, do the following:
1) Download the relevant files and save them in the "data" directory.
2) Go to "defs.py". Make sure that the relevant file name is modified, or correct if necessary.
3) The relevant dates for the mobility source files are in "mobility_model.py", lines 72-75.
4) The relevant dates for the death model are in "death_model.py", lines 54-57.

### Data sources:
Mobility data: https://www.apple.com/covid19/mobility

Death data: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series

Population size (per country): https://worldpopulationreview.com/
