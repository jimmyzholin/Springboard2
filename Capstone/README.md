# Capstone 2: Predicting Heart Disease based on Overall Health 

The National Center for Health Statistics reported that in 2020 heart disease was the leading cause of death in the US with 696,000 recorded deaths. For comparison, cancer and COVID-19 resulted in 602,000 and 350,000 deaths. Though the risk of heart disease can be decreased with healthly lifestyle choices. Through annual surveys the CDC has collected health information from US residents which we'll use to predict the likihood of heart disease given a list of self reported past diseases and other health factors.

### Objective
Develop a model that can predict a person's likihood of heart disease given that person's other health habits and metrics.

### Data
This dataset was made available on Kaggle by user Kamil Pytlak. Kamil performed some initial cleaning and feature reduction. The original dataset was complied by the Centers for Disease Control as part of its annual health survey for 2020. The survey had 400,000 adult participants and nearly 300 features.

### Exploratory Data Analysis

From EDA, we see there is a good amount of survey participants from every age group though we see the age groups above 55 nearly double some younger age groups.

![Bar graph of age groups](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/age_bar_graph.JPG)

We also see that the overwhelming majority of survey particiants identified as White.

![Bar graph of participant race](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/race_distribution.JPG)

In terms of most prominent reported health ailments, heart disease is actually fifth behind difficulty in mobility, diabetes, asthma, and skin cancer. Though since the most prominent age groups who participated are aged 55 years and above, it seems reasonable heart disease is not the highest. We also have to keep in mind that this dataset is not reporting on deaths where heart disease is ranked number one in the US.

![Bar graph of participant race](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/disease_dist.JPG)

### Modeling
To create predictions, I chose three different machine learning models: Random Forest, Support Vector Machine, and Decision Tree. 
I began with the Random Forest model for its computational speed. Also, I would not have to decide on individual feature importance as I do not necessarily known which features have the most influence on predicting heart disease. Lastly, I would not have to do extensive hyperparameter tuning with this model and so I could get results quickly to help me explore different models. For example, the results of the Random Forest helped me to decide on which features to use for the Support Vector Machine model.

![Random Forest Feature Importance Bar Graph](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/randomForestFeatureImportance.JPG)

With 18 features in this dataset and a test set of over 210,000 records, the required time to fit a Support Vector Machine was too long. After waiting hours to fit the test set to the model, I decided to narrow my test set features. I chose the top 3 relative features decided by the Random Forest model: BMI, SleepTime, and AgeCategory. With the narrowed feature test set fitting the model still took 24 minutes. Though we'll see when we compare the ROC curves, the Support Vector Machine performs the worst out of the three chosen models.

![ROC Curve: Support Vector Machine](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/ROC_SVM.JPG)

The best predictive model was the Decision Tree with a hyperparameter max depth of 5. I used Randomized Grid Cross Validation to determine the optimal max depth but I also compared the accuracy and F1 scores of several max depths.

![Decision Tree: Max Depth Comparison](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/decisionTree_max_depth_comp.JPG)

When comparing the ROC curves of all three models the Decision Tree shows the best performance and can serve as the prediction model. 

![ROC Curves of all models](https://github.com/jimmyzholin/Springboard2/blob/master/Capstone/reports/figures/ROC_Comparison.JPG)

Since the negative effects of heart disease happen over long periods of time and there is little downside to minimizing its negative effects, we can choose and tune a model that can allow more False Positive Rate so long as the True Positive Rate also improves.

### Future Exploration

We identified certain models that predict well without requiring a lot computational resources. For the future, we can continue to improve the model through hyperparameter tuning. We could also incorporate the other features of the original dataset to predict other health conditions. We could also use other models such as KNN or Hierarchy Clustering to establish groupings of participants. With groups, users of the data may be able to find patterns within certain health conditions and perhaps develop a hollistic health improvement plan. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
            └── heart_2020_preprocessed.csv <- This is file contains the preprocessed data which will be used for modeling.
            └── noNullMarketingData.csv
            └── processedUkData.csv
    │   └── raw            <- The original, immutable data dump. Several raw datasets were downloaded to discuss which would be best for the capstone project.
    │       └── heart_2020_cleaned.csv <- This is the dataset that will be used for modeling.
            └── marketing_campaign.csv
            └── uk_retail.csv
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
        └── 1.0-jl-initial-modeling.ipynb <- This contains the hyperparameter tuning, model performance, and model selection recommendation.
    │
    ├── notebooks
        └── Health_dataset_EDA.ipynb <- This is exploratory data analysis of the selected dataset for the capstone project.
        └── capstone2eda.ipynb
        └── eCommerce Behavior 2019.ipynb
        └── Preprocessing_Health_Dataset.ipynb <- This notebook contains the preprocessing steps which created the 'heart_2020_preprocessed.csv' which will be used for modeling.
        └── raw-marketing-campaign-cleaning.ipynb
        └── raw-uk-retail-cleaning.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
