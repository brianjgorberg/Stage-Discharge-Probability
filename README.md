CONTACT: Brian Gorberg  

This project was created to help organize and streamline the flood modeling process. It is used to view USGS stream gauge data, 
calculate probabilities, and determine rating curves. This is then often applied to flood models for calibration, validation, and scenario 
testing.  

In this case, we use three examples of USGS gauges in Hawaii (Waiahole, Waialua, Hanalei). We use 4 scripts to run through a 
modeling analysis. The scripts streamline the process of gathering metadata, pulling observed annual peaks, analyzing the probability, 
calculating rating curves, and then estimating model uncertainty by comparing observations to model outputs. 


**YOU MUST DOWNLOAD PEAK FQ OUTPUTS FOR THIS** ALL USGS GAUGES METADATA FOR HAWAII IS INCLUDED IN THE FOLDER 
PULL_USGS_METADATA** 
https://rconnect.usgs.gov/peakfq/ 

**YOU MUST FOLLOW THE RATING CURVE GITHUB REPOSITORY TO CREATE A KERNEL IN JUPOYTER NOTEBOOKS** 
!!!YOU SHOULD MAKE A NEW ENVIRONMENT!!! 
SEE SOURCE:  
https://github.com/thodson-usgs/ratingcurve 


The scripts do the following: 
1) Select USGS gauges to get metadata and read peakfq results 
2) Derive rating curves for the annual peaks from the peak fq results ^ 
3) Observed rating curves are cleaned into incremental stage values (1ft), the discharge is calculated from the rating curves, 
and then probabilities are estimated using the Peak Fq probability output 
4) An example is provided to compare the modeled and observed rating curves. Uncertainty in the model is calculated using 
the observed rating curves and the GSSHA flood model’s stage/discharge relationship.

Functions are found in the .py files 

SAVE_DATA and figures folders contains all script outputs 

PULL_USGS_METADATA contains Hawaii USGS metadata and a script to retrieve Hawaii gauges, but also a file of retrieved hawaii gauges 

Model_results contain the GSSHA files for script (4) 

SAVE DATA: 
USGS_Gauge_Metadata = script (1) 
Peak_Fq_USGS_probability = script (1) 
Stream_Gauge_Annual_Peaks-input = script ( 1) 
OBSERVED-rating-curves-results = script (2)  
OBSERVED-stage-discharge-probability-results = script (3) 
GSSHA-MODEL-observed-stage-uncertainty = script (4) 
GSSHA-MODEL_stage-discharge-probability-results = script (4)
