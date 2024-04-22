
# TASK 1: Classify evolution between two pairs of 2-D slices from two consecutive 2D OCT acquisitions.

--- 

This provide a guide of how to submit the this task for both the premilary phase and the final phase 

|case|prediction|
|---|---|
|1|0|
|2|3|
|3|1|
|.. | ..|
|.. | ..|
|3545 | 1|
|3546 |2 |


Where the case colunm correspond to the an unique id associed to a single example in validation dataset. The prediction colunm should only countain the class the following class:
- 0 
- 1 
- 2 
- 3 

The following test runned after we make a submission:
- The size of the .csv is equal to the number of case provided number of case
- All the id case are present
- The label should be $\in [0,3]$

We provide the submission sample, which is basicaly a .csv with two column one for case and an other one with prediction.


Submission page [Challenge website](https://mario.grand-challenge.org/submissions/)
