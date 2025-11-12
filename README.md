[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)
# IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

## 📌 Purpose of this Template

This repository is the **starter** for your End Semester Project submission in GitHub Classroom. You can implement your solution and push your work in this repository. Please free to edit this README.md file as per your requirements.

> **Scope (as per assignment brief):**
> This repository contains a report and all codes which contain solutions to multi class classification of the MNIST dataset. 
---

**Important Note:** 
1. TAs will evaluate using the `.py` file only.
2. All your reports, plots, visualizations, etc pertaining to your solution should be uploaded to this GitHub repository

---

## 📁 Repository Structure

* Describe your repository structure here. Explain about overall code organization.
  
algorithms.py : Data loading, processing and various implementations of models and ensembles from scratch trained and validated on the MNIST dataset.  

main.py : Loads final model used for classification from the algorithms.py file. To run this code, the train and test dataset must be in the same folder as the one this file is run in else the paths can be changed in the code itself.    

report.pdf : A comprehensive summary of all models tried, system architecture details, hyperparameter tuning, final hyperparameters used, training time logging, system optimizations used and finally some observations/insights from the exercise.  

xgb_hyperparam_results.csv : contains hyperparameter combinations and results for each of them on xgboost.   

---

## 📦 Installation & Dependencies

* Mention all the related instructions for installation of related packages for running your code here.

---

## ▶️ Running the Code

All experiments should be runnable from the command line **and** reproducible in the notebook.

### A. Command-line (recommended for grading)

* Mention the instructions to run you .py files.  
1. Make sure all files are in the same directory.   
2. The dataset paths in main.py have been set to "MNIST_train.csv" and "MNIST_test.csv" , if required these can be changed at the top of the main.py code.  
3. The final trained model is imported from algorithms.py in main.py code, both files should be in same directory.   
---

## You can further add your own sections/titles along with corresponding contents here:

---

## 🧾 Authors

**<Eshika Nahata, DA24B004>**, IIT Madras (2025–26)


## Best Practices:
* Keep commits with meaningful messages.
* Please do not write all code on your local machine and push everything to GitHub on the last day. The commits in GitHub should reflect how the code has evolved during the course of the assignment.
* Collaborations and discussions with other students is strictly prohibited.
* Code should be modularized and well-commented.

