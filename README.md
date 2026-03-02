Air Quality Index (AQI) Prediction Using Machine Learning

Air pollution is a major environmental and public health concern, especially in rapidly urbanizing countries like India. This project focuses on predicting the Air Quality Index (AQI) using Machine Learning techniques based on historical pollutant data collected from multiple Indian cities.

 Project Overview
 
This project analyzes approximately 30,000+ real-world data points collected from cities ranging from Ahmedabad to Visakhapatnam. The dataset includes major air pollutants such as:

	PM2.5
	PM10
	NO, NO2, NOx
	NH3
	CO
	SO2
	O3
	Benzene, Toluene, Xylene
	AQI & AQI Bucket

Using this data, we developed multiple machine learning classifiers to accurately predict AQI categories.

 Objectives
 
	Predict AQI categories using supervised machine learning algorithms
	Compare performance of different classifiers
	Identify key pollutants influencing air quality
	Provide insights to support environmental policymaking
	Enable scalable real-time AQI prediction models

 Machine Learning Models Used
 
 1. Random Forest Classifier

		Ensemble learning technique
		Handles high-dimensional data well
		Provides feature importance metrics
		Reduces overfitting

 2. Logistic Regression
    
		Linear classification model
		Probability-based prediction
		Efficient and interpretable

 3. Adaptive Boosting (AdaBoost)
    
		Ensemble boosting method
		Improves weak learners
		Focuses on difficult-to-classify samples

Project Workflow

	Data Collection
	Historical air quality data from multiple Indian cities

Data Preprocessing

	Handling missing values
	Feature cleaning and formatting
	Sub-index calculations for pollutants

Feature Engineering:

	AQI Sub-index creation
	AQI bucket classification:
		Good
		Satisfactory
		Moderate
		Poor
		Very Poor
		Severe

Model Training

	Train-test split (75/25)
	Stratified sampling
	Model fitting using sklearn

Evaluation

	Accuracy Score
	Classification Report
	Confusion Matrix
	Correlation Heatmap
	City-wise AQI visualization

Model Deployment Ready

	Models saved using Pickle (.pkl files)
	Ready for integration in web or mobile applications

 Visualizations Included
 
	 Correlation heatmap of pollutants
	 Average AQI by city (Bar chart)
	 Model performance metrics

Tech Stack

	Python
	Pandas
	NumPy
	Matplotlib
	Seaborn
	Scikit-learn
	Pickle (Model serialization)

 Project Outcomes
 
 	Accurate AQI category prediction
 	Comparative analysis of ML models
 	Identification of high-impact pollutants
 	Deployment-ready predictive models
 	Data-driven support for environmental decisions
