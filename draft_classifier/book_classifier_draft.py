import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             roc_auc_score, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_curve, 
                             precision_recall_curve,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

# HELPER FUNCTIONS
# will be used for review text cleaning 
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()    

def clean_review_text(text: str, 
               genre_words: set, 
               common_stopwords: set) -> str:
    '''
    Cleans review text by removing HTML tags, special characters, 
    digits, genre-specific words, and common stopwords.
    '''
    text = clean_text(text)
    words = text.split()
    filtered_words = [
        word for word in words 
        if word not in genre_words and word not in common_stopwords
    ]
    return ' '.join(filtered_words)

def build_tfidf(df: pd.DataFrame, 
                stop_words: List[str],
                col: str = 'text',
                max_features=300, 
                min_df=2):
    """
    Returns a content-level TF-IDF dataframe:
      columns: tf_<token>..., plus 'content_id'
    """
    vec = TfidfVectorizer(max_features=max_features, stop_words=stop_words, min_df=min_df)
    X = vec.fit_transform(df[col].fillna(''))
    tfidf_cols = [f"tf_{t}" for t in vec.get_feature_names_out()]
    #tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf_cols)
    #other considerations: svd, hspark
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(X, columns=tfidf_cols)
    tfidf_df['asin'] = df['asin'].values
    return tfidf_df, tfidf_cols

# Data Loading and Preprocessing
''' 
1. Remove junk words from reviews 
'''
# amz data loading 
amzbks_metadata = pd.read_parquet('../amz_dataset/amazon_books_metadata_sample_20k.parquet')
amzbks_reviews = pd.read_parquet('../amz_dataset/amazon_books_reviews_sample_20k.parquet')

# merge amzbooks metadata with reviews
amzbks_metadata.rename(columns={'title': 'book_title'}, inplace=True)
amzbks_reviews.rename(columns={'title': 'review_title'}, inplace=True)
amzbks = pd.merge(amzbks_reviews, amzbks_metadata, left_on='asin', right_on='parent_asin', how='inner')

# Remove duplicates based on user + book + review text
amzbks = amzbks.drop_duplicates(subset=['user_id', 'asin', 'text'], keep='first')

# data preprocessing
# removed the books with null category_sub coloumns (4092 (<1%))
amzbks = amzbks[amzbks['category_level_2_sub'].notnull()]
# removed the reviews with null review_text coloumns (111 (<1%)
amzbks = amzbks[amzbks['text'].notnull()]
amzbks = amzbks.reset_index(drop=True)

# Using their category level 2 as genre 
# sentiment analysis: Is there a correlation between review sentiment and genre? 
tqdm.pandas()
sia = SentimentIntensityAnalyzer() 
amzbks['sentiment_score'] = amzbks['text'].progress_apply(lambda x: sia.polarity_scores(x)['compound'])

# filter out genre specific words 
genre_words = set()
for genre in amzbks['category_level_2_sub'].unique():
    genre_lower = genre.lower()
    words = re.findall(r'\b[a-zA-z]+\b', genre_lower)
    genre_words.update(words)

# common stopwords list, not genre specific
common_stopwords = {
    'the', 'and', 'is', 'in', 'to', 'of', 'a', 'it', 'that', 'this', 
    'for', 'on', 'with', 'as', 'was', 'but', 'are', 'by', 'an', 
    'be', 'at', 'from', 'or', 'not', 'have', 'they', 'you', 
    'all', 'one', 'we', 'so', 'if', 'my', 'me', 'books', 'book', 'S', 'br'
}

# cleaning book review text first 
print("\nCleaning book reviews...")
amzbks['review_text'] = amzbks['text'].progress_apply(lambda x: clean_review_text(x, genre_words, common_stopwords))

# cleaning author about text 
print("\nCleaning author about text...")
amzbks['about_author'] = amzbks['author_about'].progress_apply(lambda x: clean_review_text(x, genre_words, common_stopwords) if pd.notnull(x) else '')

# cleaning book description text
print("\nCleaning book description...")
amzbks['book_desc'] = amzbks['features_text'].progress_apply(lambda x: clean_review_text(x, genre_words, common_stopwords) if pd.notnull(x) else '')

# aggregating to book level
print("\nAggregating to book level...")
bk_lvl = (
    amzbks
    .groupby(['asin', 'category_level_2_sub'], as_index=False)
    .agg({
        'sentiment_score': 'mean',       # average sentiment across reviews
        'rating': 'mean',                # average user rating for the book
        'average_rating': 'first',       
        'book_desc': 'first',            
        'about_author': 'first'          
    })
)

bk_lvl.rename(columns={'rating': 'user_rating'}, inplace=True)

# Build TF-IDF features from cleaned review text
print("\nBuilding TF-IDF features...")
# individual testing 
#tfidf_reviews, tfidf_cols = build_tfidf(amzbks, list(common_stopwords), col='review_text', max_features=300, min_df=2)
#tfidf_author, tfidf_cols = build_tfidf(amzbks, list(common_stopwords), col='about_author', max_features=200, min_df=2)
#tfidf_bookdesc, tfidf_cols = build_tfidf(amzbks, list(common_stopwords), col='book_desc', max_features=200, min_df=2)

tfidf_author, author_cols = build_tfidf(bk_lvl, list(common_stopwords), col='about_author', max_features=200, min_df=2)
tfidf_bookdesc, bookdesc_cols = build_tfidf(bk_lvl, list(common_stopwords), col='book_desc', max_features=200, min_df=2)

#concat both tfidf dataframes
all_tfidf = pd.concat([tfidf_author[author_cols], tfidf_bookdesc[bookdesc_cols]], axis=1)
tfidf_cols = author_cols + bookdesc_cols
  
print("\nBuilding features table...")
# Concat sentiment score and TF-IDF features 
features_df = pd.concat([
    bk_lvl[['asin', 'sentiment_score', 'user_rating', 'category_level_2_sub']].reset_index(drop=True),
    all_tfidf[tfidf_cols].reset_index(drop=True)
], axis=1)

# Rating features
print("\nRating features...")
features_df['user_rating'] = amzbks['rating']
features_df['book_avg_rating'] = amzbks['average_rating']
features_df['rating_deviation'] = abs(features_df['user_rating'] - features_df['book_avg_rating'])

engineered_features = [
    'sentiment_score',
    'user_rating'
]

print("\nEVALUATION OF CLASSIFIERS FOR BOOK GENRE PREDICTION\n")
# Prepare feature matrix X and target vector y
X = features_df[engineered_features + tfidf_cols]
y = features_df['category_level_2_sub']

print("\nClass distribution before filtering (book-level):")
class_counts = y.value_counts()
print(f"Total genres: {len(class_counts)}")
print(f"Genres with only 1 sample: {(class_counts == 1).sum()}")
print(f"Genres with <10 samples: {(class_counts < 10).sum()}")

min_samples = 10
valid_genres = class_counts[class_counts >= min_samples].index
mask = y.isin(valid_genres)

X_filtered = X[mask].reset_index(drop=True)
y_filtered = y[mask].reset_index(drop=True)

print(f"\nAfter filtering:")
print(f"Feature matrix shape: {X_filtered.shape}")
print(f"Target shape: {y_filtered.shape}")
print(f"Number of genres: {y_filtered.nunique()}")
print(f"Books removed: {len(X) - len(X_filtered)}")

# Show top genres
print("\nTop 10 most common genres:")
print(y_filtered.value_counts().head(10))

X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Sample data to reduce memory usage
sample_size = 50000  
if len(X_train) > sample_size:
    print(f"\nSampling {sample_size} training samples to avoid memory issues...")
    from sklearn.model_selection import train_test_split as subsample
    X_train_sampled, _, y_train_sampled, _ = subsample(
        X_train, y_train, train_size=sample_size, random_state=42, stratify=y_train
    )
    X_train = X_train_sampled
    y_train = y_train_sampled
    print(f"New train set size: {len(X_train)}")

# Define classifiers to evaluate (with reduced complexity)
classifiers = {
    'Dummy Classifier (Most Frequent)': DummyClassifier(strategy='most_frequent', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, 
                                              max_iter=2000,
                                              n_jobs=1),
    # 'Random Forest': RandomForestClassifier(
    #     random_state=42, 
    #     n_estimators=50,  
    #     max_depth=10,      
    #     min_samples_split=100,  
    #     n_jobs=1           
    # ),
    # 'Gradient Boosting': GradientBoostingClassifier(
    #     random_state=42, 
    #     n_estimators=50,  
    #     max_depth=5,       
    #     subsample=0.5      
    # ),
}

# Evaluate each classifier
results = []

for name, clf in classifiers.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics (multi-class)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Store results
    results.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report (top 5 genres):")
    top_genres = y_filtered.value_counts().head(5).index.tolist()
    print(classification_report(y_test, y_pred, labels=top_genres, zero_division=0))
    
# Summary comparison
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("SUMMARY: Classifier Comparison")
print("="*80)
print(results_df.to_string(index=False))

# Find best classifier (excluding dummy)
results_df_no_dummy = results_df[results_df['Classifier'] != 'Dummy Classifier (Most Frequent)']
best_idx = results_df_no_dummy['F1-Score'].idxmax()
best_classifier = results_df.loc[best_idx, 'Classifier']
print(f"\nBest Classifier (by F1-Score): {best_classifier}")

# Retrain best classifier on sampled data
print(f"\nRetraining {best_classifier} on sampled data...")
final_model = classifiers[best_classifier]
final_model.fit(X_train, y_train)

# Sample 10 entries from the dataset
sampled_books = amzbks.sample(n=10, random_state=42).reset_index(drop=True)

# Preprocess the sampled data
sampled_books['review_text'] = sampled_books['text'].progress_apply(
    lambda x: clean_review_text(x, genre_words, common_stopwords)
)
sampled_books['about_author'] = sampled_books['author_about'].progress_apply(
    lambda x: clean_review_text(x, genre_words, common_stopwords) if pd.notnull(x) else ''
)
sampled_books['book_desc'] = sampled_books['features_text'].progress_apply(
    lambda x: clean_review_text(x, genre_words, common_stopwords) if pd.notnull(x) else ''
)

# Build TF-IDF features for the sampled data
tfidf_author_sample, _ = build_tfidf(sampled_books, list(common_stopwords), col='about_author', max_features=200, min_df=2)
tfidf_bookdesc_sample, _ = build_tfidf(sampled_books, list(common_stopwords), col='book_desc', max_features=200, min_df=2)

# Combine TF-IDF features
all_tfidf_sample = pd.concat([tfidf_author_sample[author_cols], tfidf_bookdesc_sample[bookdesc_cols]], axis=1)

# Combine all features
features_sample = pd.concat([
    sampled_books[['asin', 'sentiment_score', 'user_rating']].reset_index(drop=True),
    all_tfidf_sample.reset_index(drop=True)
], axis=1)

# Ensure the feature matrix matches the training feature set
X_sample = features_sample[engineered_features + tfidf_cols]

# Predict genres using the trained model
predicted_genres_sample = final_model.predict(X_sample)

# Add predictions back to the sampled data
sampled_books['predicted_genre'] = predicted_genres_sample

# Display the results
print(sampled_books[['asin', 'predicted_genre']])

# # FEATURE IMPORTANCE ANALYSIS FOR LOGISTIC REGRESSION
# print("\nAnalyzing feature importance for Logistic Regression...")
# if best_classifier == 'Logistic Regression':
#     feature_names = engineered_features + tfidf_cols
#     coefs = final_model.coef_
#     class_labels = final_model.classes_
     
#     # Option 1: Show overall feature importance (across all genres)
#     print("\n" + "="*80)
#     print("OVERALL FEATURE IMPORTANCE (averaged across all genres)")
#     print("="*80)
#     avg_abs_coef = np.abs(coefs).mean(axis=0)
#     top_overall_indices = avg_abs_coef.argsort()[::-1]

#     print(f"Length of feature_names: {len(feature_names)}")
#     print(f"Length of avg_abs_coef: {len(avg_abs_coef)}")
#     print(f"Top overall indices: {top_overall_indices}")
    
#     print("\nTop 20 most important features (by average absolute coefficient):")
#     for rank, idx in enumerate(top_overall_indices):
#         print(f"{rank:2d}. {feature_names[idx]:30s}: {avg_abs_coef[idx]:.4f}")
    
#     # Option 2: Show top features for selected genres only
#     print("\n" + "="*80)
#     print("TOP FEATURES PER GENRE (for most common genres)")
#     print("="*80)
#     top_genres = y_filtered.value_counts().head(10).index.tolist()
        
#     for class_label in top_genres:
#         if class_label in class_labels:
#             i = list(class_labels).index(class_label)
#             coef_class = coefs[i]
#             sorted_indices = coef_class.argsort()[::-1]
            
#             print(f"\nTop 10 features for '{class_label}':")
#             for rank, idx in enumerate(sorted_indices[:10], 1):
#                 print(f"  {rank:2d}. {feature_names[idx]:30s}: {coef_class[idx]:7.4f}")

#     # for i, class_label in enumerate(class_labels):
#     #     coef_class = coefs[i]
#     #     top_positive_indices = coef_class.argsort()[-10:][::-1]
#     #     top_negative_indices = coef_class.argsort()[:10]
        
#     #     print(f"\nTop positive features for genre '{class_label}':")
#     #     for idx in top_positive_indices:
#     #         print(f"  {feature_names[idx]}: {coef_class[idx]:.4f}")
        
#     #     print(f"\nTop negative features for genre '{class_label}':")
#     #     for idx in top_negative_indices:
#     #         print(f"  {feature_names[idx]}: {coef_class[idx]:.4f}")