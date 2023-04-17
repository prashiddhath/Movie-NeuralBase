from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from neural_search.config import features_weight


def construct_tfidf_plot(df: pd.DataFrame) -> Tuple[List, List]:
    """
    Creates Term Frequency–Inverse Document Frequency (TFID) vectors based entirely
    on the movie plots, without english stop words such as 'the' and 'a', for the
    given dataframe (df).

    Parameters
    -------
    df: pandas.DataFrame
        Dataframe with tmdb movie data.

    Returns
    -------
    vectors: list
        TF-IDF vectors based on movie plot.

    payload: list
        Movie titles corresponding to the TF-IDF vectors which act as identifiers.

    """
    tfidf = TfidfVectorizer(stop_words="english")

    df["overview"] = df["overview"].fillna("")

    tfidf_matrix = tfidf.fit_transform(df["overview"])

    vectors = tfidf_matrix.A
    payload = [{"title": title} for title in df["title"]]

    return vectors, payload


def create_metadata_soup(movie_entry: pd.core.series.Series) -> str:
    """
    For a movie, it combines multiple weighted feature from the featues_weight
    in config.py to a variable called soup. This soup can be then converted to
    vectors using desired embedding.

    The weight of the features can adjusted, for example, if the weight for di-
    rector is increased from 1 then, when viewing a certain movie, the resulting
    movie recommendations are more likely to be from the same director.

    Parameters
    -------
    movie_entry: pandas.core.series.Series
        A movie entry with all the features from the pandas DataFrame.

    Returns
    -------
    soup: str
        A combination of multiple weighted movie features which can be embedded
        to vectors.

    """
    soup = " "

    for feature, weight in features_weight.items():
        for _ in range(weight):
            if movie_entry[feature] is np.nan:
                continue
            soup += " ".join(movie_entry[feature])

    return soup


def construct_metadata_vectors(df: pd.DataFrame) -> Tuple[List, List]:
    """
    Creates Count vectors, for the given dataframe, based on metadata soup
    which is dervied from a combination of multiple weighted movie features
    such as director, cast, and genres.

    Parameters
    -------
    df: pandas.DataFrame
        Dataframe with tmdb movie data.

    Returns
    -------
    vectors: list
        Count vectors based on movie plot.

    payload: list
        Movie titles corresponding to the Count vectors which act as identifiers.

    """
    df["soup"] = df.apply(create_metadata_soup, axis=1)
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df["soup"])

    vectors = count_matrix.A

    payload = [{"title": title} for title in df["title"]]

    # For some reason, the Qdrant cluster doesn't allow vectors in int
    return vectors.astype(float), payload


def construct_title_vectors(model: SentenceTransformer, titles: List) -> List:
    """
    Embeds all the movie titles, provided in the list, to vectors using the ML model
    specified in config.py.

     Parameters
    -------
    model: sentence_transformers.SentenceTransformer
        Pre-trained machine learning model for embedding

    titles: list
        Movie titles to be embedded as vectors.

    Returns
    -------
    vectors: list
        Embedded vectors based on movie title.

    """
    vectors = []
    batch_size = 64
    batch = []

    for title in titles:
        batch.append(title.lower())
        if len(batch) >= batch_size:
            vectors.append(model.encode(batch))
            batch = []

    if len(batch) > 0:
        vectors.append(model.encode(batch))
        batch = []

    vectors = np.concatenate(vectors)

    return vectors
