# BT5153 - GROUP 7 - FAKE REVIEWS DETECTION

In recent years, online consumer reviews have gained increasing importance and have become a fundamental aspect of the shopping experience and decision making for customers across e-commerce and traditional retail sectors. The rise in fake reviews driven by their profitability has led to a growing presence of deceptive feedback, posing risks to both consumers and businesses. Consequently, identifying these fake reviews is essential to protect consumers and honest businesses. In this project, we explore various deep learning models using an Amazon e-commerce dataset containing both GPT-2 generated fake reviews and genuine product reviews to develop a fake review detector. We train, validate, and refine the model using this dataset to improve its performance. Our experiments show that the deep learning model we propose effectively detects fake reviews, achieving impressive performance metrics including precision, recall, and F1-score, thereby demonstrating its state-of-the-art efficacy.

## Code
- `BT5153_EDA.ipynb` - Performs class distribution analyses and statistical analysis for the review texts of our dataset
- `kaggle feature extraction.ipynb` - Computes Textblob subjectivity and polarity scores and performs Part of Speech tagging analysis
- `BT5153_Logistic Regression.ipynb` - Constructs the logistic regression baseline model 
- `BT5153_DISTILBERT.ipynb` - Constructs the pretrained distilbert model from the Hugging Face Transformers library
- `BT5153_Distilroberta-base.ipynb` - Constructs the pretrained distilroberta model from the Hugging Face Transformers library
- `NEW_BT5153_Multi-head attention.ipynb` - Constructs the self-trained multi-head attention model and generates prediction results using the kaggle review testing dataset and the independent dataset 
- `kaggle ground truth POS.ipynb` - Performs subjectivity, polarity, and part of speech tagging analyses for our kaggle groundtruth data
- `kaggle model prediction POS.ipynb` - Performs subjectivity, polarity, and part of speech tagging analyses based on the model prediction results from the our kaggle dataset
- `independent dataset ground truth POS.ipynb` - Performs subjectivity, polarity, and part of speech tagging analyses for our groundturh independent dataset
- `independent dataset predicted POS.ipynb` - Performs subjectivity, polarity, and part of speech tagging analyses for the model prediction results from the independent dataset
- `BT5153 Group7 Independent Dataset Generation.ipynb` - Generate our own fake review dataset using 'AmazonKindle_realReviews.csv'

## Data
- `fake_reviews_dataset.csv` - The kaggle fake reviews dataset for model development (Source: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- Independent Data Set
  - `IndependentReviewData.csv` - The independent review dataset we self-generated with 50 rows only
  - `IndependentReviewData_150.csv` - The independent review dataset we self-generated with 150 rows
  - 'AmazonKindle_realReviews.zip': - The dataset we used to generate our own fake reviews for model testing 
- Prediction Results
  - `Predicted_Classes_Independent_dataset_MINLOSS.csv` - The prediction results based on the independent dataset with our self-generated fake reviews 
  - `Predicted_Classes_New_MINLOSS.csv`- The prediction results based on the testing set of our kaggle fake review dataset
  
  
  


