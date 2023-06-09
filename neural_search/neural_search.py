from neural_search.prepare_data import load_movie_data
from neural_search.metric import (
    construct_tfidf_plot,
    construct_metadata_vectors,
)
from neural_search.config import (
    model,
    tfidf_coll_name,
    titles_coll_name,
    metadata_coll_name,
)
from neural_search.upload import establish_conn

from typing import List, Optional


class NeuralSearch:
    def __init__(self):
        self._df = load_movie_data()
        self._no_movies = self._df.shape[0]

        self._qdrant_client = establish_conn()

        self._tfidf_coll_name = tfidf_coll_name
        self._metadata_coll_name = metadata_coll_name
        self._titles_coll_name = titles_coll_name

        self._vectors_tfidf, _ = construct_tfidf_plot(self._df)
        self._vectors_metadata, _ = construct_metadata_vectors(self._df)

        self._model = model

    @property
    def vectors(self):
        return self._vectors

    @property
    def qdrant_client(self):
        return self._qdrant_client

    def get_movie_index(self, movie_title: str) -> int:
        """
        Returns the index (row count) of a movie.

        """
        idx = self._df.index[self._df["title"] == movie_title].to_list()
        if len(idx) == 1:
            return idx[0]

    def get_movie_vector_tfidf(self, movie_title: str) -> List:
        """
        Returns the plot-based TF-IDF vector of a movie.

        """
        idx = self.get_movie_index(movie_title)
        if idx is not None:
            return self._vectors_tfidf[idx]

    def get_movie_vector_metadata(self, movie_title: str) -> List:
        """
        Returns the metadata-based Count vector of a movie.

        """
        idx = self.get_movie_index(movie_title)
        if idx is not None:
            return self._vectors_metadata[idx]

    def get_random_movie_titles(self, n: Optional[int] = 4) -> List:
        """
        Returns n random movie titles. It is used to suggest random movie
        titles under search bar in home page.

        """
        random_entries = self._df.sample(n=n)
        return random_entries["title"].values

    def movie_exists(self, movie_title: str) -> bool:
        """
        Checks if a movie exists in our NeuralBase.

        """
        idx = self.get_movie_index(movie_title)

        if idx:
            return True

        return False

    def get_movie_overview(self, movie_title) -> str:
        """
        Returns the plot (overview) of a movie.

        """
        idx = self.get_movie_index(movie_title)
        description = self._df.iloc[[idx]]["overview"].values[0]
        return description

    def get_movie_genres(self, movie_title: str) -> List:
        """
        Returns the genres of a movie

        """
        idx = self.get_movie_index(movie_title)
        try:
            genres = self._df.iloc[[idx]]["genres"].values[0]
        except:
            return []

        genres = [genre.capitalize() for genre in genres]
        return genres

    def search_movies(self, query: str):
        """
        For a given query, it searches for the closest matching movies
        in the 'titles' vector space.

        """
        vector = self._model.encode(query.lower()).tolist()

        search_result = self._qdrant_client.search(
            collection_name=self._titles_coll_name,
            query_vector=vector,
            query_filter=None,
            top=5,
        )

        payloads = [hit.payload["title"] for hit in search_result]
        return payloads

    def recommend_movies(self, movie_title: str, type: str) -> List:
        """
        For a movie in the database, it recommends similar movies based either
        on the plot (tf-idf) or metadata (count).

        """
        if not self.movie_exists(movie_title):
            return []

        vector = []

        if type == "tfidf":
            collection_name = self._tfidf_coll_name
            vector = self.get_movie_vector_tfidf(movie_title)
        elif type == "count":
            collection_name = self._metadata_coll_name
            vector = self.get_movie_vector_metadata(movie_title)
        else:
            return []

        search_result = self._qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=None,
            top=5,
        )
        payloads = [hit.payload["title"] for hit in search_result]
        return payloads[1:]
