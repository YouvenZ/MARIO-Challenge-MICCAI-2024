# TASK 2: Prediction of evolution within 3 mounts of AMD on OCT 2D slices for planning treatment anti-VEGF.


This provide a guide of how to submit the this task for both the premilary phase and the final phase 

|case|prediction|
|---|---|
|1|0|
|2|0|
|3|1|
|.. | ..|
|.. | ..|
|3545 | 1|
|3546 |2 |

The case id and the label prediction are random, was provided to present the structure of the csv expected. Please follow the case id that will be given once you have acces to the dataset. 


Where the case colunm correspond to the an unique id associed to a single example in the validation dataset. The prediction colunm should only contain  the following class:
- 0 
- 1 
- 2 

The following test runned after we make a submission:
- The size of the .csv is equal to the number of case provided number of case
- All the id case are present
- The label should be $\in [0,2]$

We provide the submission sample, which is basicaly a .csv with two column one for case and an other one with prediction.


Submission page [Challenge website](https://www.codabench.org/competitions/2851/)
