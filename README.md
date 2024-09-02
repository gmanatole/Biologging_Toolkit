# BIOLOGGING TOOLKIT

This package aims to handle biologging datasets, from raw data processing to specific applications on acoustic or inertial data.
Most processing functions automatically access DTAG4 data, but classes also handle raw inputs.

It is divided into different sections :
- processing : Creates finalized dataset of processed data (sound pressure level data, animal posture and heading, jerk data, etc.) from raw data.
- applications : Uses the finalized dataset for specific use cases (wind estimation from acoustics, prey-catch attempts, drift dive detections, etc.).
- utils : Contains specific python functions used in the different modules.
- auxiliary : Python codes to download auxiliary data (sun position, all ERA5-reanalysis data)
- plot : Functions called in notebooks for interactive plotting

This package is meant to grow throughout the years, so please don't hesitate to contribute !
The main objective is to provide general codes with associated notebooks so that different users can have the same tools and parameters for their data analysis.


## How to install the package

Download the package and unzip it or run the following command :

`git clone https://github.com/gmanatole/SES_tags.git`

Then install the necessary packages or run :

`pip -r requirements.txt`
