import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
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
                col: str = 'text',
                max_features=300, 
                min_df=2):
    """
    Returns a content-level TF-IDF dataframe:
      columns: tf_<token>..., plus 'content_id'
    """
    vec = TfidfVectorizer(max_features=max_features, stop_words='english', min_df=min_df)
    X = vec.fit_transform(df[col].fillna(''))
    tfidf_cols = [f"tf_{t}" for t in vec.get_feature_names_out()]
    tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf_cols)
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

# Build TF-IDF features from cleaned review text
print("\nBuilding TF-IDF features...")
tfidf_df, tfidf_cols = build_tfidf(amzbks, col='review_text', max_features=300, min_df=2)

print("\nBuilding features table...")
# Concat sentiment score and TF-IDF features 
features_df = pd.concat([
    amzbks[['asin', 'sentiment_score', 'category_level_2_sub']].reset_index(drop=True),
    tfidf_df[tfidf_cols].reset_index(drop=True)
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
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check class distribution
print("\nClass distribution before filtering:")
class_counts = y.value_counts()
print(f"Total genres: {len(class_counts)}")
print(f"Genres with only 1 sample: {(class_counts == 1).sum()}")
print(f"Genres with <10 samples: {(class_counts < 10).sum()}")

# Filter out genres with fewer than 2 samples bc of stratified split
min_samples = 10  # trying at least 10 samples per genre
valid_genres = class_counts[class_counts >= min_samples].index
print(f"\nFiltering to genres with at least {min_samples} samples...")

# Filter both X and y
mask = y.isin(valid_genres)
X_filtered = X[mask].reset_index(drop=True)
y_filtered = y[mask].reset_index(drop=True)

print(f"\nAfter filtering:")
print(f"Feature matrix shape: {X_filtered.shape}")
print(f"Target shape: {y_filtered.shape}")
print(f"Number of genres: {y_filtered.nunique()}")
print(f"Samples removed: {len(X) - len(X_filtered)} ({(len(X) - len(X_filtered))/len(X)*100:.2f}%)")

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
