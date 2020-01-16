============
### Stellar Mass Prediction
### Building multiple regression model to predict the mass of the stars
============

by: Anastasia Gorina and Andres Chaves

multiple regression analysis

=========================================================================================

    ├── Data                 <- The original and cleaned data used in this project,
    |
    ├── Notebooks            <- Jupyter notebooks.
    |
    ├── Reports              <- Generated analysis as HTML, PDF, Slides, etc.
    |
    ├── README.md            <- The top-level README for developers using this project.
    |
    └── src                  <- Source code for use in this project.




=========================================================================================
NASA's search for exoplanets began over 30 years ago, and the exploration of the stars started even earlier in history. Today NASA has information about over 4000 exoplanets and their host stars. Some of the values are observed and measured (directly or indirectly), some are calculated. The goal of this project was to use NASA's dataset to predict the stellar mass. 

Obtaining data:
NASA has multiple API's https://api.nasa.gov/. We used The Exoplanet Archive API that has the information about all confirmed exoplanets and their stars.

Cleaning:
Unfortunately, the table turned out to be incomplete, it had a lot of missing values. We dropped the rows with NaNs and normalized the values by applying the log-transformation.

Analysis and Testing:
The goal was to build a multiple regression model with the highest possible R^2 and minimal error. The input (independent) variables were stellar temperature, radius, and luminocity. The output (dependent) variable was stellar mass - that is what we were trying to predict. We ended up with the best model with R^2 of 0.796 and minimal error (on both train and test subsets) after adding polynomial terms and Ridge regularization. 
