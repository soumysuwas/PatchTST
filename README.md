# PatchTST

This repo has whole repo content of 3 open source github repos, created by using the : "Gittodoc" program ([https://www.gittodoc.com/](https://www.gittodoc.com/)), a tool to convert Git repositories into documentation link that you can feed to AI coding tools like cursor.

Following repos have been added here :

[Original_repo.txt](Original_repo.txt) : From the : https://github.com/yuqinie98/PatchTST

[neuralforecast_full_repo.txt](neuralforecast_full_repo.txt) : From the : https://github.com/Nixtla/neuralforecast

[tsai_whole_repo.txt](tsai_whole_repo.txt) : From the : https://github.com/timeseriesAI/tsai

So now like i need to create a new code

 i am a PhD student and my research focuses on landslide prediction, in past one year i prepared landslide susceptibility mapping using static factors and dynamic factors like rainfall and soil moisture and made yearwise dynamic lsm. now next i have to prepare future year forecasted lsm using forecasted rainfall and for this i need to forecast the rainfall for the further year. i have past 11 years rainfall 5km grid size data to forecast future year grid wise rainfall.
 
 So I need to forecast the further next years rainfall.
 
 above i have attached the repos that has implementation guide for the PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
 
 the timeseriesAI/tsai and Nixtla/neuralforecast have implemented and included and created a framework that uses PatchTST.
 
 I want to use PatchTST for forecasting of rainfall

 the rainfall data i have is in csv files in a grided format, let me explain about it a bit, here:
 
 there are total 11 csv files all having same structure
 
 the files are named as like this: '2014ProcessedMeanRainfallGriddedData5000m.csv' to '2024ProcessedMeanRainfallGriddedData5000m.csv' , only changinf the value of year from2014 to 2024
 the structure is like this: 
 
 RangeIndex: 32485 entries, 0 to 32484
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   system:index  32485 non-null  object 
 1   Date          32485 non-null  object 
 2   Latitude      32485 non-null  float64
 3   Left_Lon      32485 non-null  float64
 4   Longitude     32485 non-null  float64
 5   Lower_Lat     32485 non-null  float64
 6   Rainfall      32485 non-null  float64
 7   Right_Lon     32485 non-null  float64
 8   Upper_Lat     32485 non-null  float64
 9   .geo          32485 non-null  object 
dtypes: float64(7), object(3)
memory usage: 2.5+ MB
None
system:index	Date	Latitude	Left_Lon	Longitude	Lower_Lat	Rainfall	Right_Lon	Upper_Lat	.geo
0	0_0	01-01-2014	11.473421	76.121721	76.144179	11.450963	0.000000	76.166637	11.495878	{"geodesic":false,"type":"Point","coordinates"...
1	0_1	02-01-2014	11.473421	76.121721	76.144179	11.450963	0.000000	76.166637	11.495878	{"geodesic":false,"type":"Point","coordinates"...
2	0_2	03-01-2014	11.473421	76.121721	76.144179	11.450963	0.207457	76.166637	11.495878	{"geodesic":false,"type":"Point","coordinates"...
3	0_3	04-01-2014	11.473421	76.121721	76.144179	11.450963	3.398571	76.166637	11.495878	{"geodesic":false,"type":"Point","coordinates"...
4	0_4	05-01-2014	11.473421	76.121721	76.144179	11.450963	3.087930	76.166637	11.495878	{"geodesic":false,"type":"Point","coordinates"...

where in the column named 'system:index' we have the reference for the Grid number and day number , the values are like this: 0_0, 0_1, .....,0_364/365 ,1_0, 1_1, .....1_364/365, ......88_364              

here the values are representing the grid number and day number such that the value before '_' is the grid number and the value after is day number , for example in 1_5 the 1 is the grid number and the 5 is the day number reperesenting the 5th day of the year, the value of grid number goes from 0 to 88 totalling to 89 grids and the value for day number goes from 0 to 364 or 365 depending on whether it is leap year or not representing the days of the years.

next the column named 'Date' has the date in the format DD-MM-YYYY for each corresponding system:index 

next the column named 'Rainfall' has the value of rainfall in mm for that day

these are 3 main and relevant columns : 'system:index', 'Date', 'Rainfall'
 
i want to write a code that can predict/forecast rainfall with least errors, get inspiration and ideas and code architecture as said above , and write the code for me , also also when doing so , make it so that the code works both on GPU and CPU if available , also include any installs for dependecy for pip, the code i going to be run primarily on the kaggle or colab, i want to have the grid by grid prediction, and for checking if evything is working correctly only do prediction for one grid, give me code , thanks
