---
logo_width: 70
logo_height: 20

python_logo: "<img src=\"https://logos-marcas.com/wp-content/uploads/2021/10/Python-Logo.png\" alt=\"Python Logo\" style=\"width: 75px; height: 20px; object-fit: cover; object-position: center;\">"
powerbi_logo: "<img src=\"https://www.jeveuxetredatascientist.fr/wp-content/uploads/2023/09/power-bi-vector-logo-2022.jpg\" alt=\"PowerBI Logo\" style=\"width: 75px; height: 20px; object-fit: cover; object-position: center;\">"
numpy_logo: '<img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy Logo" width="70" height="20">'
pandas_logo: '<img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" alt="Pandas Logo" width="70" height="20">'
pingouin_logo: '<img src="https://pingouin-stats.org/build/html/_images/logo_pingouin.png" alt="Pingouin Logo" width="70" height="20">'
scipy_logo: '<img src="https://lense.institutoptique.fr/ressources/MINE/Tutoriels/Python/MINE_Python_Scipy_Logo.png" alt="SciPy Logo" width="70" height="20">'
seaborn_logo: '<img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" alt="Seaborn Logo" width="70" height="20">'
matplot_logo: '<img src="https://matplotlib.org/stable/_images/sphx_glr_logos2_003.png" alt="Matplotlib Logo" width="70" height="20">'
plotly_logo: '<img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png" alt="Plotly Logo" width="70" height="20">'
sklearn_logo: '<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn Logo" width="70" height="20">'
xgboost_logo: '<img src="https://upload.wikimedia.org/wikipedia/commons/5/58/XGBoost_logo.svg" alt="XGBoost Logo" width="70" height="20">'
lgbm_logo: '<img src="https://upload.wikimedia.org/wikipedia/commons/d/d9/LightGBM_logo_black_text.svg" alt="LightGBM Logo" width="70" height="20">'
catboost_logo: '<img src="https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/When-to-Choose-CatBoost-Over-XGBoost-or-LightGBM-Practical-Guide_8.png?ssl=1" alt="CatBoost Logo" width="70" height="20">'
surprise_logo: '<img src="images/logo.png" alt="Surprise Logo" width="70" height="20">'
tensorflow_logo: '<img src="https://www.freelogovectors.net/wp-content/uploads/2018/07/tensorflow-logo.png" alt="Tensorflow Logo" width="70" height="20">'
beautifulsoup_logo: "<img src=\"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fhwk0702.github.io%2Fimg%2Fbs.png&f=1&nofb=1&ipt=e6bebc477a0128c41176abc2df088f6e79d18654df9d16288258e25a8e20dc36\" alt=\"Beautifulsoup Logo\" style=\"width: 70px; height: 20px; object-fit: cover; object-position: center;\">"
---

# Selected Portfolio Projects

<div class="dhdelimiter">
<hr>
</div>

## <span style="color: #000080;">Python Projects</span> {{ page.python_logo}}

### Italy University Enrollments
In this project, I’ve acquired data from the MIUR (Italy’s Ministry of Education, University and Research) to delve into the differences in university enrollments in Italy from 2000 to 2020.

<img src="images/Italy un.png" />

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.pandas_logo }} 
    {{ page.seaborn_logo }}
  </div>
</div>

---

### Sales Analysis and Customer Segmentation
In order to get as much information as possible about a sales dataset, analysis techniques such as EDA, inferential analysis, and ML were used. Customer segmentation practices are also employed to group similar users, providing the possibility of targeting customers in the best way possible.

<img src="images/Sales_Analysis.png" alt="Sales Analysis" />

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.pandas_logo }}
    {{ page.matplot_logo }}
    {{ page.seaborn_logo }}
    {{ page.plotly_logo }}
    {{ page.scipy_logo }}
    {{ page.pingouin_logo }}
    {{ page.sklearn_logo }}
    {{ page.xgboost_logo }}
    {{ page.catboost_logo }}
    {{ page.lgbm_logo }}
  </div>
</div>

[View code on Colab](https://colab.research.google.com/drive/16D2wZIYugOM2u7OUHegozT25UXVQB6gV?usp=sharing){:target="_blank"}

---

### Movie Recommendation System
This notebook delves into recommendation systems capable of making movie recommendations using collaborative filtering. The models tested are a Tensorflow deep model with embedded layers and the well-known SVD algorithm.

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.tensorflow_logo }}
    {{ page.surprise_logo }}
  </div>
</div>

[View code on Colab](https://colab.research.google.com/drive/1NrYWsHU4AvId_r-SOSqpcFeX7kO8I70O?usp=sharing){:target="_blank"}

---

### Temperatures Analysis and Forecasting
To gain insights into the temperature change across the world, I cleaned and analyzed a table containing the average temperatures of major cities worldwide. The cleaned table is then used to find out if there is any association between the temperatures set and the CO2 emissions using a second table. After the analysis of the association, the dataset is finally used to develop a model that can predict the temperature on a given day and location.

<img src="images/World_Temperature.png" alt="World Temperatures" />

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.numpy_logo }}
    {{ page.pandas_logo }}
    {{ page.matplot_logo }}
    {{ page.seaborn_logo }}
    {{ page.beautifulsoup_logo }}
    {{ page.matplot_logo }}
    {{ page.sklearn_logo }}
    {{ page.xgboost_logo }}
    {{ page.tensorflow_logo }}
  </div>
</div>
[View code on Colab](https://colab.research.google.com/drive/1H3_I_27C7ZFKb_ZGP_X_CAr97t-C9qtk?usp=sharing){:target="_blank"}

---

### Animal Image Classifier
In this notebook, I utilized Tensorflow to build a model capable of predicting the name of an animal present in an image. To accomplish this, I decided to use the fast and high-performing InceptionV3 architecture. 

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.tensorflow_logo }}
  </div>
</div>

[View code on Colab](https://colab.research.google.com/drive/13ZDnaX4-0tyE3VQM_HZ-7czKSJ5jiET8?usp=sharing){:target="_blank"}

---

### NLP for Fake News Detection
Fake news detection is the task of ascertaining the authenticity of news articles and determining whether they are genuine or fake. In this project, I utilized Tensorflow to develop a model capable of predicting the probability of news articles being fake. The model is then used to create a web app classifier using Streamlit.

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.tensorflow_logo }}
  </div>
</div>

- [View code on Colab](https://colab.research.google.com/drive/12jPrkVDvf1OSUcSKqdgly9Sr5Yyz9nx-?usp=sharing){:target="_blank"}
- [View web app on Streamlit](https://newsclassifier-ffhsmr5htxfjrj3jqyaayd.streamlit.app/){:target="_blank"}

---

### NLP for Text Summarization and Sentiment Analysis
In this notebook, I will go through the steps of building an attention-based sequence-to-sequence model using the encoder-decoder architecture with TensorFlow. The model will take a text sequence as input and output the corresponding summary and an auxiliary sentiment analysis output.  

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.tensorflow_logo }}
  </div>
</div>

[View code on Colab](https://colab.research.google.com/drive/1KUvyLiefE3sZiqr9-t3erUCcJLcbwafC?usp=sharing){:target="_blank"}

---

### ISIC 2024 - Skin Cancer Detection
Annual competition held on Kaggle aimed at the diagnosis of skin cancer through the use of images and tabular data. I participated by developing computer vision models for image analysis, subsequently integrating their outputs with tabular data to enhance the performance of the final predictive models obtaining a partial area under the ROC curve (pAUC) above 80% true positive rate of 0.15683.  

<img src="images/isic_2024.png" alt="ISIC 2024 Skin Cancer Detection" />

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.tensorflow_logo }}
  </div>
</div>


[View Kaggle competition](https://www.kaggle.com/competitions/isic-2024-challenge){:target="_blank"}

---

### BrisT1D Blood Glucose Prediction
Time series competition held on Kaggle that had the goal of predicting blood glucose fluctuations in individuals with type 1 diabetes. To tackle the problem presented I experimented with both boosting algorithms and neural networks but continued only with Convolutional Neural Networks (CNNs) as they significantly outperformed the other models, managing to obtain an RMSE score of 2.4635.  

<img src="images/brist1d.png" alt="BrisT1D Blood Glucose Prediction" />

<div class="logos-background-full">
  <div class="logos-background">
    {{ page.sklearn_logo }}
    {{ page.xgboost_logo }}
    {{ page.catboost_logo }}
    {{ page.lgbm_logo }}
    {{ page.tensorflow_logo }}
    </div>
</div>

[View Kaggle competition](https://www.kaggle.com/competitions/brist1d){:target="_blank"}

---

### Kaggle Playground Series


| Episode | Task | Score |
|---------|------|-------|
| S3E15 | Feature Imputation with a Heat Flux | RMSE: 0.081516 |
| S3E22 | Predict Health Outcomes of Horses | Micro F1: 0.76829 |
| S4E2  | Multi-Class Prediction of Obesity Risk | Accuracy: 0.90769 |
| S4E5  | Regression with Flood Prediction | R²: 0.86620 |
| S4E6  | Classification with Academic Success | Accuracy: 0.83787 |
| S4E9  | Regression of Used Car Prices | RMSE: 63,637 |

<div class="dhdelimiter">
<hr>
</div>

## <span style="color: #000080;">Power BI Reports</span> {{ page.powerbi_logo }}

### Flight Analisys

This project investigates the relationship between coach prices and various flight characteristics.

<img src="images/powerbi/flight analisys.png" alt="flight analisys" />

### Income Analisys

This project investigates the relationship between Income and various characteristics.

<img src="images/powerbi/income analysis_Pagina_1.png" alt="income analisys 01" />
<img src="images/powerbi/income analysis_2.png" alt="income analisys 02" />
