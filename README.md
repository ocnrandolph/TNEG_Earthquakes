## Data Question 4: Web Scraping Earthquake Data

### Part 1:
Nate Silver discusses the difficulty of predicting earthquakes in **The Signal and the Noise**. Nevertheless, we will try to identify some patterns by analyzing the deadly earthquakes that have occurred since 1900.

To start, read the table of earthquakes from https://en.wikipedia.org/wiki/List_of_deadly_earthquakes_since_1900 using the `requests` and/or `beautifulsoup` library and load it to a pandas dataframe. You will need to do some data cleaning before you can proceed.

Data cleaning tasks include:

* Replace empty strings with NaN
* Remove the footnotes from the 'Other Source Deaths' column
* Convert Magnitude to a numeric type. For this portion, you can ignore differences in seismic magnitude scales.
* Correct number of deaths when there is more than one value. When there is more than one value given, choose the largest.
* Create a new column ('deaths') that evaluates the four total-death columns ('PDE Total Deaths', 'Utsu Total Deaths', 'EM-DAT Total Deaths', and 'Other Source Deaths') and populates the new column with the highest value.
* Explore the data in terms of when and where earthquakes occurred and how severe they were (magnitude, deaths, secondary effects).

Also, add any supplemental data you'd like in order to explore ideas related to earthquake occurrence and effects, but understand that it is not required.

Answer the following questions:

1. Are there factors that make an earthquake more likely?
2. Are there factors that make an earthquake more deadly?
 
### Part 2:
Use the `requests` library and the USGS's API (https://earthquake.usgs.gov/fdsnws/event/1/) to retrieve information about all recorded earthquakes that occurred in Tennessee since 1900.
Answer the following questions:

1. How many earthquakes are in the dataset? Which region of Tennessee seems to be the most prone to earthquakes? Which is the least prone?
2. Do the magnitudes of earthquakes seem to follow the distribution described by Nate Silver. That is, one higher magnitude is ten times less likely to occur?

### Part 3:
Based on your findings from Parts 1 and 2, put together a presentation for the Tennessee Earthquake Preparedness Group to help them answer the following two questions:
1. Is it advisable to concentrate our resources in the four major metropolitan areas (Memphis, Nashville, Chattanooga, Knoxville)?

2. How should we allocate resources amongst these areas?
    * Which area is likely to experience an earthquake?
    * Which area is likely to suffer loss from an earthquake if one hits?