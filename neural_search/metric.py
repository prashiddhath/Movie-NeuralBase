from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from .config import features_weight


def construct_tfidf_plot(df):
    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words="english")

    # Replace NaN with an empty string
    df["overview"] = df["overview"].fillna("")

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df["overview"])

    vectors = tfidf_matrix.A
    payload = [{"title": title} for title in df["title"]]

    return vectors, payload


def create_metadata_soup(data):
    soup = " "

    for feature, weight in features_weight.items():
        for _ in range(weight):
            if data[feature] is np.nan:
                continue
            soup += " ".join(data[feature])

    return soup


def construct_metadata_vectors(df):
    df["soup"] = df.apply(create_metadata_soup, axis=1)
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df["soup"])

    vectors = count_matrix.A

    payload = [{"title": title} for title in df["title"]]

    # Need to change vectors to float, int vectors not upload somehow
    return vectors.astype(float), payload


def construct_title_vectors(model, titles):
    vectors = []
    batch_size = 64
    batch = []

    for title in titles:
        batch.append(title)
        if len(batch) >= batch_size:
            vectors.append(model.encode(batch))  # Text -> vector encoding happens here
            batch = []

    if len(batch) > 0:
        vectors.append(model.encode(batch))
        batch = []

    vectors = np.concatenate(vectors)

    return vectors
