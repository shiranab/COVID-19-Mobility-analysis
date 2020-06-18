# COVID-19-Mobility-analysis
## Analysis for our manuscript: COVID 19 pandemic related lockdown: response time is more important than its strictness

### See our paper: https://www.medrxiv.org/content/10.1101/2020.06.11.20128520v1

### Authors: Gil Loewenthal, Shiran Abadi, Oren Avram, Keren Halabi, Noa Ecker, Natan Nagar, Itay Mayrose, and Tal Pupko

The rapid spread of SARS-CoV-2 and its threat to health systems worldwide have led governments to take acute actions to enforce social distancing. Previous studies used complex epidemiological models to quantify the effect of lockdown policies on infection rates. However, these rely on prior assumptions or on official regulations. Here, we use country-specific reports of daily mobility from people cellular usage to model social distancing. Our data-driven model enabled the extraction of mobility characteristics which were crossed with observed mortality rates to show that: (1) the time at which social distancing was initiated is of utmost importance and explains 62% of the number of deaths, while the lockdown strictness or its duration are not as informative; (2) a delay of 7.49 days in initiating social distancing would double the number of deaths; and (3) the expected time from infection to fatality is 25.75 days and significantly varies among countries.


The attached code fits the proposed models to the mobility data (Apple) and the daily deaths data (John's Hopkins University), extracts the relevant features, and runs the correlation between each feature and the fit of the regression analysis (paper, figure 4).
