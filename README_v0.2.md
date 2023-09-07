# Project Overview: KKBOX
WSDM - KKBox's Churn Prediction Challenge: Can we predict when subscribers will churn?
Details about the project: https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data

KKBox, founded in 2005 in Taiwan, is the major music streaming service in Asia. With a vast collection of over 40 million licensed tracks, KKBox has garnered a substantial user base of over 10 million across Taiwan, Hong Kong, Japan, Singapore, and Malaysia.

In this challenge, my task is to predict whether a user will discontinue their subscription after it expires. More precisely, the goal is to forecast whether a user will make a new service subscription transaction within 30 days after their current membership expiration date.

## 1.Problem Area
### Startup Index
- The "Rule of 40" is a crucial financial benchmark for startups. It's computed by adding the firm's revenue growth rate to its profit margin, with Monthly Recurring Revenue (MRR) acting as a key indicator of this growth rate. However, MRR isn't determined by just one aspect. It is instead analyzed from multiple angles, including new business acquisition, current customer behavior, and churn rates.

### Better Business Performance
- Startups, particularly app-based businesses operating on subscription models like music streaming services - the source of our dataset for this project - often face customer churn as one of their most pressing challenges. The ability to predict customer churn is paramount, considering that retaining existing customers usually comes at a lower cost than acquiring new ones.

- The early identification of churn risk customers provides startups with the opportunity to proactively engage with potentially departing customers, thus improving retention. This key metric is especially critical for startups seeking funding, as investors often scrutinize churn rates to evaluate the startup's viability and growth potential.

## 2.Proposed Data Science Solution
Given the structure of the data, the solution will be a classification problem, specifically customer churn prediction. In this scenario, the goal is to predict whether a customer will stop using a service based on their behavior and transaction history.

There are three impportant points I have to take into account:  
1: The size of the data is relatively large, necessitating consideration of analytical load.  
2: The dataset spans a wide range of categories, from customer attribute data to time-series data and usage status data.  
3: The data has been properly assigned with IDs, allowing for individual customer analysis.  

Based on these perspective, I would suggest following solutions for classification:
Logistic Regression: This is a simple and efficient model for binary classification problems. It's especially useful when fast, interpretable model and your features have a linear relationship with the log odds of the outcome. However, the data may be too complex and logistic regression might not be the best choice.

Decision Trees and Random Forests: Decision trees are also interpretable and can model non-linear relationships. Random forests, which are ensembles of decision trees, are even more powerful and can reduce some of the overfitting problems that individual decision trees have. However, they can be slower and more computationally intensive than logistic regression.

## 3.Impact of the Solution
Business Implication: Churn analysis improves customer satisfaction, loyalty, and ultimately, revenues, and most notably MRR as a company health score.  
  Retention Strategies: It aids in devising effective customer retention strategies.  
  Resource Allocation: It guides optimal allocation of resources for maximum retention impact.  
  Customer Segmentation: It enables targeted customer segmentation for personalized marketing and retention efforts.  

IT Implication: Churn analysis drives IT strategies for better system design and customer service improvement.  
  Product Enhancement: It informs product development with customer retention in mind.  
  Data-Driven Decision Making: It enables decisions based on data, improving strategy effectiveness.  
  CDP (Customer Data Platform): It supports the use of a CDP for a unified view of customer behavior, aiding retention efforts.  

## 4.Dataset Description
The data required for this project is available at https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data. 
Please note that the CSV files are not included in this repository at this point due to the file size.

### Data Dictionary
| No. | Category | Column Name | Description | Concerns |
|-----|----------|-------------|-------------|----------|
| 1 | User Identification | msno | Unique ID for each user. | n/a |
| 2 | Target Variable | is_churn | Indicates whether a user has churned or not. | n/a |
| 3 | Transaction Info | payment_method_id | The method used for payment. | n/a |
| 4 | Transaction Info | payment_plan_days | The length of the membership plan. | Need to check the unrealistic value 0. Unsubscribed? |
| 5 | Transaction Info | plan_list_price | The list price of the plan. | Need to check the unrealistic value 2000. |
| 6 | Transaction Info | actual_amount_paid | The amount actually paid for the plan. | Need to check the unrealistic value 0. Unpaid? |
| 7 | Transaction Info | is_auto_renew | Indicates whether the plan renews automatically. | n/a |
| 8 | Transaction Info | is_cancel | Indicates whether the user canceled the membership in a transaction. | n/a |
| 9 | User Engagement | date | Date of user log. | n/a |
| 10 | User Engagement | num_25, num_50, num_75, num_985, num_100 | Songs played less than XX% of the song length. | n/a |
| 11 | User Engagement | num_unq | Number of unique songs played. | n/a |
| 12 | User Engagement | total_secs | Total seconds of music played. | n/a |
| 13 | Demographic Info | city | Same as the column names. | n/a |
| 14 | Demographic Info | bd (age) | Same as the column names. | Outlier values ranging from -7000 to 2015. Insert 0 or average bd to these numbers? |
| 15 | Demographic Info | gender | Same as the column names. | 65% of data are null-value. Drop the column or change the column name (unknown) and leave? |
| 16 | Demographic Info | registered_via | Registration method. | Outlier value -1.00 |
| 17 | Demographic Info | registration_init_time | The time of registration. | n/a |


## 5.EDA
- df_members:
city: Majority from cities 1 and 4.
bd: Contains erroneous age values.
registered_via: Contains anomalous values like -1.
Visualizations highlight:
Dominance of City 1.
Majority registered in 2016 and 2017.

- df_train:
is_churn: Majority haven't churned.

- df_trans:
Some columns suggest potential free or missing data.
membership_expire_date contains unrealistic maximum values.
Visualizations highlight:
Predominance of Payment Method 41.
Majority have auto-renewal active.

- df_userlog:
Accurate date range.
Potential outliers in song listening columns.
Visualizations show right-skewed distributions after log transformation.


## 6.Data Cleaning and Preparation

1. Preparation for Data Merger:
Made a copy of df_trans as df_trans2.
Converted the membership_expire_date to datetime format.
Retained the latest membership_expire_date for each 'msno'.
Checked and confirmed no duplicate msno entries.
Removed the date column from df_userlog, calculated the sum of each column per msno, and counted occurrences of each msno.

2. Data Merger:
Merged df_members, df_trans2, and df_userlog_sum to form train_dataset and test_dataset.
Checked for missing values and data types in each dataset.

3. Data Preprocessing:
Handled outlies, null-value, data type conversion, one-hot encoding.

## Feature Engineering
In the process of enhancing our model's prediction capabilities, we have introduced several new features derived from existing data columns. The new features aim to encapsulate customer behavior metrics that are integral to calculating the Lifetime Value (LTV) of a customer. 

The LTV is formulated considering three major perspectives - Average Order Value, Purchase Frequency (or Engagement), and Average Customer Lifespan. The developed features are categorized based on these perspectives:

1. Average Order Value
1.1. is_discount: This binary feature indicates whether a discount was applied on a transaction, with 1 denoting that the actual_amount_paid is less than the plan_list_price, and 0 otherwise.

1.2. discount_amount: This feature calculates the discount amount for each transaction as the difference between plan_list_price and actual_amount_paid.

2. Purchase Frequency (Engagement)
Features in this category aim to quantify the engagement level of users with the service.

2.1. Average Play Time per Song: This feature represents the average time a user spends on listening to a song. It is derived by dividing the total seconds played by the total number of songs played.

2.2. Full Play Rate: This metric denotes the proportion of songs played over 98.5% of their length to the total number of songs played, giving insight into a user's tendency to listen to songs in full.

2.3. Unique Song Play Rate: Indicates the proportion of unique songs played to the total number of songs played, which can help identify users' appetite for diversity in their music selection.

2.4. Skip Tendency: A metric showing the tendency of a user to skip songs before they reach 25% of their length. It is derived from the ratio of 'num_25' to the total number of songs played.

2.5. Repeat Tendency: This metric denotes the tendency of a user to repeat songs, calculated as the difference between the total number of songs played and the number of unique songs played, divided by the total number of songs played.

3. Average Customer Lifespan
3.1. Membership Period in Days: This feature calculates the number of days between the membership_expire_date and the transaction_date, representing the length of the customer's membership period.

### Total list of the features used for modeling

 #   Column                       Non-Null Count   Dtype  
---  ------                       --------------   -----  
 0   msno                         725722 non-null  object 
 1   is_churn                     725722 non-null  int64  
 2   bd                           725722 non-null  int64  
 3   registration_init_time       725722 non-null  float64
 4   payment_plan_days            725722 non-null  int64  
 5   actual_amount_paid           725722 non-null  float64
 6   is_auto_renew                725722 non-null  int64  
 7   is_cancel                    725722 non-null  int64  
 8   msno_count                   725722 non-null  float64
 9   city_agg_4.0                 725722 non-null  int64  
 10  city_agg_5.0                 725722 non-null  int64  
 11  city_agg_13.0                725722 non-null  int64  
 12  city_agg_15.0                725722 non-null  int64  
 13  city_agg_22.0                725722 non-null  int64  
 14  city_agg_Other               725722 non-null  int64  
 15  gender_1                     725722 non-null  int64  
 16  gender_2                     725722 non-null  int64  
 17  registered_via_4.0           725722 non-null  int64  
 18  registered_via_9.0           725722 non-null  int64  
 19  registered_via_13.0          725722 non-null  int64  
 20  payment_method_id_agg_38.0   725722 non-null  int64  
 21  payment_method_id_agg_39.0   725722 non-null  int64  
 22  payment_method_id_agg_40.0   725722 non-null  int64  
 23  payment_method_id_agg_41.0   725722 non-null  int64  
 24  payment_method_id_agg_Other  725722 non-null  int64  
 25  is_discount                  725722 non-null  int64  
 26  discount_amount              725722 non-null  float64
 27  avg_play_time                725722 non-null  float64
 28  full_play_rate               725722 non-null  float64
 29  skip_tendency                725722 non-null  float64
 30  repeat_tendency              725722 non-null  float64
 31  membership_period            725722 non-null  int64  


## Modeling:
In our churn prediction project, we have devised a three-step strategy to select the optimal model. Here’s a brief overview of each step:

1. Baseline Model - Logistic Regression
As a starting point, a logistic regression model will be used to establish a performance benchmark. It's chosen for its simplicity and effectiveness in binary classification scenarios.

2. Scalable Models - SGD Classifier and XGBoost (GBDT)
Given the large dataset, we progress to scalable models - the SGD Classifier and XGBoost. These models efficiently handle larger datasets and may incorporate upsampled data to address the 'is_churn' class imbalance, enhancing the performance.

3. Decision Tree and Random Forest
Finally, we'll explore tree-based models: Decision Tree and Random Forest. These models, while potentially slower, offer insight into feature importance and may employ downsampling to balance computation time and accuracy.

## Model Evaluation

### Evaluation Metrics
To assess the performance of our churn prediction models, we rely on several metrics. Each offers a distinct perspective:

Accuracy: While indicative of overall correctness, its reliability diminishes with imbalanced data.
Recall: Particularly vital here, given the high cost associated with misclassifying churn customers.
F1 Score: Assists in maintaining a balance between Precision and Recall, beneficial for our case.
ROC-AUC: A crucial metric that evaluates the distinction capabilities of the classifier between classes, especially with our imbalanced data.
In this churn prediction scenario, ROC-AUC, Recall, and F1 score are believed to be the most pertinent evaluation tools due to the nature of our data and business priorities.

### Model Performance
Here’s how each model performed on different metrics:

Model	ROC-AUC	Recall	F1 Score
Logistic Regression	0.76	0.52	0.66
SGD Classifier	0.82	0.74	0.48
XGBoost	0.79	0.74	0.26
Decision Tree	0.84	0.70	0.65
Random Forest	0.96	0.89	0.89

### Analysis
Random Forest: Exhibited signs of overfitting, and due to computational constraints, utilizing unsampled data was not feasible.
Logistic Regression: Despite a decent F1 score, the low recall for class 1 indicates a deficiency in correctly identifying churn customers.
Decision Tree: Presents a balanced performance, with high values in ROC, Recall, and F1 scores, indicating an effective approach in churn prediction.
Considering our goals, it seems a balanced approach as demonstrated by the Decision Tree model might be the optimal choice for this case.

## Implication

### Coefficiency of Logistic Regression
- payment_plan_days: The likelihood of churn increases as the duration of the payment plan becomes longer. This is because customers who opt for longer-term plans are more likely to churn within that extended period, influenced by the commitment associated with such plans.

- is_auto_renew: When auto-renewal is enabled, the probability of churn decreases. Auto-renewal is a convenient feature for customers, eliminating the need for manual renewal and thereby enhancing the likelihood of continued subscription.

- is_cancel: If cancellations occur, the probability of churn rises. Cancellations can act as precursors to churn, thereby contributing to an increased likelihood of customers leaving the service.

### Decision Tree
From the decision tree analysis, it became evident that factors such as the membership period, the presence of cancellation (is_cancel), and a specific payment method (payment_method_41) are correlated with the churn rate.