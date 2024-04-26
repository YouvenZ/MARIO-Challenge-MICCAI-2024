# 🕹️ 🍄 MARIO : Monitoring AMD progression in OCT

![](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/mario_banner_.png)


[Challenge website](https://youvenz.github.io/MARIO_challenge.github.io/)

## Context

This is the github page of the MARIO challenge. This page aims to provide guidance of how to publish your solution to codabench.

We provide the following information

- The script we used for the evaluation
- A docker exemple for the submission for final phase 

---


## Tasks

### Task 1 

The first task focuses on pairs of 2D slices (B-scans) from two consecutive OCT acquisitions. The goal is to classify the evolution between these two slices (before and after), which clinicians typically examine side by side on their screens.

![](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/mario_task_1_gray_bg.png)

### Task 2

The second task focuses on 2D slices level. The goal is to predict the future evolution within 3 months with close monitoring of patients that are enrolled in an anti-VEGF treatments plan. In summary, the first task aims to automate the initial step of the analysis (useful for decision support) and the second task aims to automate the complete analysis process (useful for autonomous AI).

![](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/mario_task_2_gray_bg.png)


## Submission instruction

### How to register to the challenges

![](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/tuto_register.png)

### How to have acces to the data ?

First if you do not have a CodaBench account yet, create one [here](https://www.codabench.org/accounts/signup): 

Second fill the following [form](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/MARIO%202024%20Data%20Challenge%20Participation%20Form.pdf), fill the form and send it to rachid.zeghlache@univ-brest.fr. Also please make sure to add "MARIO 2024 Challenge" in the subject of your mail in order to faciliate the. Finally add the your Team name for naming form (i.e  MARIO 2024 Data Challenge Participation Form_team_name.pdf). Once completed and signed, we will verify the document and within the next 48 working hours you will receive the a link to download the dataset. 


Our challenge use the two stack (.csv submission and algorithm submission)                                                     

- For the premilary phase you need to submit your .csv prediction of the validation set

- For the final phase would required all succeful team to submit a container of their code to the the same adress. Using the as subject "Container solution [Team_name]". More details will be disclose later on. 

🎯 Note that only teams (or single participant) with a valid submission for both tasks will be considered for final ranking and thus the rewards.



