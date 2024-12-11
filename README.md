# CPSC322 Final Project: Near Earth Object Classifier

All of the classiifications and the final report are in NEO_Classification.ipynb   

Link to our classifier API, hosted on Render:   
https://cpsc322-finalproject-0itg.onrender.com/predict?estimated_diameter_min=0&estimated_diameter_max=1&relative_velocity=2&miss_distance=1   
To make a prediction, users will need to input the attribute values into the URL so that it goes predict?attribute=value&attribute=value... 

NOTE: Values should be discretized before being inputted   
Discretization key:   
    `estimated_diameter_min`: 1 = 0.0 - 0.1; 2 = 0.1 - 0.2; 3 = 0.2 - 1.1   
    `estimated_diameter_max`: 1 = 0.0 - 0.2; 2 = 0.2 - 0.4; 3 = 0.4 - 2.5   
    `relative_velocity`: 1 = 6199.1 - 43077.5; 2 = 43077.5 - 64028.3; 3 = 64028.3 - 138171.4   
    `miss_distance`: 1 = 131164.3 - 31876675.5; 2 = 31876675.5 - 54714412.0; 3 = 54714412.0 - 74715777.4   
If values are not within the discretized bins, they can be put in the closest bin, but these values may not provide accurate classification results.  

NOTE: Render may run slowly so please have patience as it generates the results of the classification   