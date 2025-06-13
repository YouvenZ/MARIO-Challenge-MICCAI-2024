Of course, here is a visually improved version of your README file with the added citation information.

# 🕹️ 🍄 MARIO: Monitoring AMD Progression in OCT

**[Challenge Website](https://youvenz.github.io/MARIO_challenge.github.io/)**

-----

## 📝 About the Challenge

Welcome to the official GitHub repository for the **MARIO Challenge**\! This page provides all the necessary information and resources to help you successfully participate and submit your solution to Codabench.

Here, you will find:

  * The evaluation script we use.
  * A Docker example for your final phase submission.

-----

## 🎯 Tasks

### **Task 1: Pair-wise B-scan Classification**

This task focuses on analyzing pairs of 2D slices (B-scans) from two consecutive OCT acquisitions. Your goal is to classify the evolution between these two slices (before and after), mimicking how clinicians examine them side-by-side.

### **Task 2: Future Evolution Prediction**

This task challenges you to predict the future evolution of 2D slices within a 3-month timeframe for patients undergoing anti-VEGF treatment. This task aims to automate the entire analysis process, paving the way for autonomous AI in clinical decision-making.

-----

## 🚀 Submission Instructions

### **How to Register**

### **Accessing the Data**

1.  **Create a Codabench Account:** If you don't have one, [sign up here](https://www.codabench.org/accounts/signup).
2.  **Complete the Participation Form:** Download the [form here](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/MARIO%202024%20Data%20Challenge%20Participation%20Form.pdf), fill it out, and email it to `rachid.zeghlache@univ-brest.fr`.
      * **Email Subject:** "MARIO 2024 Challenge"
      * **File Name:** `MARIO 2024 Data Challenge Participation Form_team_name.pdf`
3.  **Receive the Data:** Once we verify your submission (within 48 working hours), we will send you a link to download the dataset.

### **Submission Phases**

Our challenge has two submission phases:

  * **Preliminary Phase:** Submit your `.csv` predictions for the validation set.
  * **Final Phase:** Successful teams must submit a container of their code to the same email address.
      * **Email Subject:** "Container solution [Team\_name]"

> **🎯 Important:** Only participants with valid submissions for both tasks will be considered for the final ranking and rewards.

-----

## 📜 Citation

If you use this work, please cite the following paper:

```bibtex
@misc{zeghlache2025deeplearningretinaldegeneration,
      title={Deep Learning for Retinal Degeneration Assessment: A Comprehensive Analysis of the MARIO AMD Progression Challenge}, 
      author={Rachid Zeghlache and Ikram Brahim and Pierre-Henri Conze and Mathieu Lamard and Mohammed El Amine Lazouni and Zineb Aziza Elaouaber and Leila Ryma Lazouni and Christopher Nielsen and Ahmad O. Ahsan and Matthias Wilms and Nils D. Forkert and Lovre Antonio Budimir and Ivana Matovinović and Donik Vršnak and Sven Lončarić and Philippe Zhang and Weili Jiang and Yihao Li and Yiding Hao and Markus Frohmann and Patrick Binder and Marcel Huber and Taha Emre and Teresa Finisterra Araújo and Marzieh Oghbaie and Hrvoje Bogunović and Amerens A. Bekkers and Nina M. van Liebergen and Hugo J. Kuijf and Abdul Qayyum and Moona Mazher and Steven A. Niederer and Alberto J. Beltrán-Carrero and Juan J. Gómez-Valverde and Javier Torresano-Rodríquez and Álvaro Caballero-Sastre and María J. Ledesma Carbayo and Yosuke Yamagishi and Yi Ding and Robin Peretzke and Alexandra Ertl and Maximilian Fischer and Jessica Kächele and Sofiane Zehar and Karim Boukli Hacene and Thomas Monfort and Béatrice Cochener and Mostafa El Habib Daho and Anas-Alexis Benyoussef and Gwenolé Quellec},
      year={2025},
      eprint={2506.02976},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.02976}, 
}
```
