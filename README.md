# 🕹️ 🍄 MARIO: Monitoring AMD Progression in OCT

[![arXiv](https://img.shields.io/badge/arXiv-2506.02976-b31b1b.svg)](https://arxiv.org/abs/2506.02976)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/15270469)
[![MICCAI 2024](https://img.shields.io/badge/MICCAI-2024-green)](https://youvenz.github.io/MARIO_challenge.github.io/)

<p align="center">
  <img src="https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/mario_banner_.png" alt="MARIO Challenge Banner" width="800">
</p>

## 📋 Overview

This is the official repository for the MARIO (Monitoring AMD progression in OCT) Challenge. Here you'll find guidance on how to participate and submit your solutions to Codabench.

**[Challenge Website](https://youvenz.github.io/MARIO_challenge.github.io/)**

### Resources Provided:

- Evaluation scripts
- Docker example for final phase submissions
- Access to the [dataset on Zenodo](https://zenodo.org/records/15270469)

---

## 🎯 Challenge Tasks

### Task 1: Comparative Analysis

<p align="center">
  <img src="https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/mario_task_1_gray_bg.png" alt="Task 1" width="700">
</p>

The first task focuses on pairs of 2D slices (B-scans) from two consecutive OCT acquisitions. The goal is to classify the evolution between these two slices (before and after), which clinicians typically examine side by side on their screens.

### Task 2: Future Progression Prediction

<p align="center">
  <img src="https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/mario_task_2_gray_bg.png" alt="Task 2" width="700">
</p>

The second task operates at the 2D slice level. The goal is to predict the future evolution within 3 months for patients enrolled in anti-VEGF treatment plans. While Task 1 aims to automate the initial analysis step (decision support), Task 2 aims to automate the complete analysis process (autonomous AI).

> 🔔 **Note**: Only teams with valid submissions for both tasks will be considered for final ranking and rewards.

---

## 📝 Participation Guide

### How to Register

<p align="center">
  <img src="https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/images/tuto_register.png" alt="Registration Tutorial">
</p>

### Accessing the Dataset

1. Create a [CodaBench account](https://www.codabench.org/accounts/signup) if you don't have one
2. Download and complete the [participation form](https://github.com/YouvenZ/MARIO-Challenge-MICCAI-2024/blob/main/MARIO%202024%20Data%20Challenge%20Participation%20Form.pdf)
3. Send the form to rachid.zeghlache@univ-brest.fr with "MARIO 2024 Challenge" in the subject
4. Name your form as "MARIO 2024 Data Challenge Participation Form_team_name.pdf"
5. Once verified, you'll receive a download link within 48 working hours
6. Alternatively, access the [public dataset on Zenodo](https://zenodo.org/records/15270469)

### Submission Process

- **Preliminary Phase**: Submit your CSV predictions for the validation set
- **Final Phase**: Submit a container of your code to the same address with subject "Container solution [Team_name]"

---

## 📚 Citation

If you use the MARIO challenge dataset or reference our work, please cite:

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
