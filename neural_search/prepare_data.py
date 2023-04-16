import numpy as np
import pandas as pd

from neural_search.config import movies_csv, credits_csv
from ast import literal_eval


def load_movie_data():
    fields = ["genres", "title", "id", "keywords", "overview", "production_companies"]

    df_movies = pd.read_csv(movies_csv, skipinitialspace=True, usecols=fields)

    fields = ["movie_id", "cast", "crew"]
    df_credits = pd.read_csv(credits_csv, skipinitialspace=True, usecols=fields)

    df_credits.columns = ["id", "cast", "crew"]
    df_movies = df_movies.merge(df_credits, on="id")

    df_movies = prepare_data(df_movies)

    return df_movies


def prepare_data(df):
    nested_features = ["cast", "crew", "keywords", "genres", "production_companies"]
    df = perform_literal_eval(df, nested_features)

    df["director"] = df["crew"].apply(get_director)

    features = ["genres", "keywords", "cast", "production_companies"]
    df = reduce_features_data(df, features)
    df = standardize_data(df, features)

    return df


def standardize_data(df, features):
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


def perform_literal_eval(df, features):
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    return df


def reduce_features_data(df, features, n=4):
    for feature in features:
        df[feature] = df[feature].apply(get_top_entries, n=n)

    return df


def get_top_entries(entries, n):
    if isinstance(entries, list):
        top_entries = []

        for i, entry in enumerate(entries):
            if i >= n:
                break
            top_entries.append(entry["name"])

        return top_entries

    return []


def get_director(crews):
    for crew_data in crews:
        if crew_data["job"] == "Director":
            return crew_data["name"]
    return np.nan
