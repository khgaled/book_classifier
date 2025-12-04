import pandas as pd
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)
from sklearn.linear_model import LogisticRegression
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

print("\nBuilding concatenated review text per book...")
reviews_concat = (
    amzbks
    .groupby('asin')['review_text']
    .apply(lambda s: " ".join(s.astype(str)))
    .reindex(bk_lvl['asin'])
    .fillna("")
    .tolist()
)

# Build SBERT features from cleaned review text
print("\nBuilding SBERT features...")
model = SentenceTransformer("all-mpnet-base-v2")

cache_path = Path("emb_cache.npz")

if cache_path.exists():
    print("\nLoading embeddings from cache...")
    data = np.load(cache_path, allow_pickle=True)
    review_emb = data["review_emb"]
    author_emb = data["author_emb"]
    bkdesc_emb = data["bkdesc_emb"]
    cached_asin = data["asin"]
    
    # safety check: make sure cache matches current books
    if len(cached_asin) != len(bk_lvl) or not np.array_equal(cached_asin, bk_lvl["asin"].values):
        print("Cache does not match current bk_lvl; recomputing embeddings...")
        cache_path.unlink()  # delete bad cache
        # fall through to recompute below
        cache_path = None
else:
    cache_path = None

if cache_path is None:
    print("\nComputing embeddings (this may take a while)...")

    print("\nBook review embeddings...")
    review_emb = model.encode(reviews_concat, show_progress_bar=True)

    print("\nAuthor review embeddings...")
    author_emb = model.encode(bk_lvl['about_author'].tolist(), show_progress_bar=True)

    print("\nBook Desc embeddings...")
    bkdesc_emb = model.encode(bk_lvl['book_desc'].tolist(), show_progress_bar=True)

    np.savez(
        "emb_cache.npz",
        asin=bk_lvl["asin"].values,
        review_emb=review_emb,
        author_emb=author_emb,
        bkdesc_emb=bkdesc_emb,
    )
    print("Saved embeddings to emb_cache.npz")

#pool review embeddings at book level
review_emb_bklvl = review_emb 
bert_features = np.hstack([review_emb_bklvl, author_emb, bkdesc_emb])

#column names similar to tf-idf tokens
author_cols = [f"author_emb_{i}" for i in range(author_emb.shape[1])]
bkdesc_cols = [f"bkdesc_emb_{i}" for i in range(bkdesc_emb.shape[1])]
review_cols = [f"review_emb_{i}" for i in range(review_emb_bklvl.shape[1])]

sbert_cols = review_cols + author_cols + bkdesc_cols

sbert_df = pd.DataFrame(bert_features, columns=sbert_cols)
sbert_df['asin'] = bk_lvl['asin'].values

print("\nBuilding features table...")
# Concat sentiment score and TF-IDF features 
features_df = pd.concat([
    bk_lvl[['asin', 'sentiment_score', 'user_rating', 'category_level_2_sub']].reset_index(drop=True),
    sbert_df[sbert_cols].reset_index(drop=True)
], axis=1)

engineered_features = [
    'sentiment_score',
    'user_rating'
]

print("\nEVALUATION OF CLASSIFIERS FOR BOOK GENRE PREDICTION\n")
# Prepare feature matrix X and target vector y
X = features_df[engineered_features + sbert_cols]
y = features_df['category_level_2_sub']

print("\nClass distribution before filtering (book-level):")
class_counts = y.value_counts()
print(class_counts.head(10))

min_samples = 10
valid_genres = class_counts[class_counts >= min_samples].index
mask = y.isin(valid_genres)

X_filtered = X[mask].reset_index(drop=True)
y_filtered = y[mask].reset_index(drop=True)

print(f"\nAfter filtering (book-level):")
print("Feature matrix shape:", X_filtered.shape)
print("Target shape:", y_filtered.shape)
print("Number of genres:", y_filtered.nunique())
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

#num_classes = y_filtered.nunique()

# Define classifiers to evaluate (with reduced complexity)
classifiers = {
    'Dummy Classifier (Most Frequent)': DummyClassifier(strategy='most_frequent', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, 
                                              max_iter=2000,
                                              n_jobs=1)
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

# used for further testing and model 
def predict_genre_for_new_book(reviews,
                               book_desc: str,
                               about_author: str,
                               user_rating=None,
                               genre_words=genre_words,
                               common_stopwords=common_stopwords,
                               sia=sia,
                               sbert_model=model,
                               final_model=final_model,
                               engineered_features=engineered_features,
                               sbert_cols=sbert_cols,
                               bk_lvl=bk_lvl):
    """
    Predict genre for a new books not in the dataset.
    
    reviews: list of raw review strings (can be 1+)
    book_desc: raw book description (string)
    about_author: raw author bio (string)
    user_rating: numeric (float/int), e.g. 4.3. If None, uses dataset mean.
    """

    if isinstance(reviews, str):
        reviews = [reviews]

    # sentiment score calculation
    if len(reviews) == 0:
        sentiment_score = 0.0
        concat_reviews = ""
    else:
        sentiment_scores = [sia.polarity_scores(r)['compound'] for r in reviews]
        sentiment_score = float(np.mean(sentiment_scores))

        # clean and concatenate reviews 
        cleaned_reviews = [
            clean_review_text(r, genre_words, common_stopwords) for r in reviews
        ]
        concat_reviews = " ".join(cleaned_reviews)

    # user_rating calculation
    if user_rating is None:
        user_rating = float(bk_lvl['user_rating'].mean())

    # Clean description and author text
    desc_clean = clean_review_text(str(book_desc), genre_words, common_stopwords)
    author_clean = clean_review_text(str(about_author), genre_words, common_stopwords)

    # SBERT embeddings: review text, author, description
    review_vec = sbert_model.encode([concat_reviews])[0]
    author_vec = sbert_model.encode([author_clean])[0]
    desc_vec   = sbert_model.encode([desc_clean])[0]

    sbert_feature_vec = np.hstack([review_vec, author_vec, desc_vec])

    # Combine engineered + SBERT 
    all_features = np.hstack([[sentiment_score, user_rating], sbert_feature_vec])

    all_feature_names = engineered_features + sbert_cols
    X_single = pd.DataFrame([all_features], columns=all_feature_names)

    pred_label = final_model.predict(X_single)[0]
    pred_proba = final_model.predict_proba(X_single)[0]

    return pred_label, pred_proba