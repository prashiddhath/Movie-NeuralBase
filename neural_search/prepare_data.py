import numpy as np
import pandas as pd

from neural_search.config import movies_csv, credits_csv, max_data
from ast import literal_eval

from typing import List, Optional, Union


def load_movie_data() -> pd.DataFrame:
    """
    Reads the relevant data from the "tmdb_5000_movies.csv" and "tmdb_5000_credits.csv" files,
    pre-processes it, and stores the information into a pandas dataframe.

    Using the max_data parameters from config.py, it is possible to specify the number of rows
    to be read from the csv files. The free-tier Qdrant cluster cannot store all the data req-
    uired for this application, so the deployed version sets the max_data parameter to 3800.

    Returns
    -------
    df_movies: pandas.DataFrame
        Dataframe with pre-processed movie information which can be used to build desired feat-
        ure vectors.
    """

    fields_movies = [
        "genres",
        "title",
        "id",
        "keywords",
        "overview",
        "production_companies",
    ]
    fields_credits = ["movie_id", "cast", "crew"]

    if max_data:
        df_movies = pd.read_csv(
            movies_csv, skipinitialspace=True, usecols=fields_movies, nrows=max_data
        )
        df_credits = pd.read_csv(
            credits_csv, skipinitialspace=True, usecols=fields_credits, nrows=max_data
        )
    else:
        df_movies = pd.read_csv(
            movies_csv, skipinitialspace=True, usecols=fields_movies
        )
        df_credits = pd.read_csv(
            credits_csv, skipinitialspace=True, usecols=fields_credits
        )

    df_credits.columns = ["id", "cast", "crew"]
    df_movies = df_movies.merge(df_credits, on="id")

    df_movies = prepare_data(df_movies)

    return df_movies


def prepare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    As the raw data in df_raw also consists of Python literals and not only string, all the
    parsed information are first evaluated for literals, followed by reducing large features
    , such as cast and crew, to limited entries, default to 4 items. Finally, the data is
    standardized by converting all entries to lower space and removing the white spaces.

    Parameters
    -------
    df_raw: pandas.DataFrame
        Dataframe with raw data parsed from the tmdb csv files.

    Returns
    -------
    df_final: pandas.DataFrame
        Dataframe with pre-processed movie information which can be used to build desired feat-
        ure vectors.
    """

    nested_features = ["cast", "crew", "keywords", "genres", "production_companies"]
    df = perform_literal_eval(df_raw, nested_features)

    # Extract director name from the crew feature and create a separate director column
    df["director"] = df["crew"].apply(get_director)

    features = ["genres", "keywords", "cast", "production_companies"]
    df = reduce_features_data(df, features)

    df_final = standardize_data(df, features)

    return df_final


def standardize_data(df: pd.DataFrame, features: List) -> pd.DataFrame:
    """
    For the given feature set, it standardized all the movie data such that there
    are no white spaces and everything is stored in lower case.

    Parameters
    -------
    df: pandas.DataFrame
        Dataframe with tmdb movie data.

    features: list
        List of data features (column) in the provided dataframe that is to be standardized.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with standardize data for the given movie features.

    """

    def clean_data(data):
        if isinstance(data, list):
            return [str.lower(item.replace(" ", "")) for item in data]
        else:
            if isinstance(data, str):
                return str.lower(data.replace(" ", ""))

        return ""

    for feature in features:
        df[feature] = df[feature].apply(clean_data)

    return df


def perform_literal_eval(df: pd.DataFrame, features: List) -> pd.DataFrame:
    """
    For the given feature set, it evaluates for Python literals such as dictionaries, in the
    csv files that are otherwise only parsed as raw strings.

    Parameters
    -------
    df: pandas.DataFrame
        Dataframe with tmdb movie data.

    features: list
        List of data features (column) in the provided dataframe that is to be evaluated for
        literals.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with literals for the given movie features.

    """
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    return df


def reduce_features_data(
    df: pd.DataFrame, features: List, n: Optional[int] = 4
) -> pd.DataFrame:
    """
    For each given feature in a movie entry, it reduces the total number of items to
    be a maximum size of n and stores it as a list.

    For example, the "tmdb_5000_credits.csv" consists of crews and casts feature where
    the entries are stored as list of dictionary and could easily exceed 50 entries.
    For such cases, the function only picks the top n entries (names) to be stored as
    list.

    Parameters
    -------
    df: pandas.DataFrame
        Dataframe with tmdb movie data.

    features: list
        List of data features (column) in the provided dataframe that is to be reduced

    n: int, optinal
        Maximum number of feature items to be stored for each movie entry

    Returns
    -------
    df: pd.DataFrame
        Dataframe with reduced data as list for the given movie features.

    """
    for feature in features:
        df[feature] = df[feature].apply(get_top_entries, n=n)

    return df


def get_top_entries(feature_data: List, n: int) -> List:
    """
    For a given feature data (entries) of each movie, it iterates over a maximum of n first
    items, appends the name to a list and returns it.

    """
    if isinstance(feature_data, list):
        top_entries = []

        for i, entry in enumerate(feature_data):
            if i >= n:
                break
            top_entries.append(entry["name"])

        return top_entries

    return []


def get_director(crews: List):
    """
    From the crews feature of each movie, it finds the name of the Director and, if available,
    returns it.

    """
    for crew in crews:
        if crew["job"] == "Director":
            return crew["name"]
    return np.nan
