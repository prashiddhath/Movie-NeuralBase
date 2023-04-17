# Movie NeuralBase

## Introduction
Movie NeuralBase is an AI-based web-app that can be used to search for movies. When a movie is selected, the app also shows its relevant metadata and movie recommendations similar to it. The web-app utilizes `Qdrant` engine to store embedded vectors and perform search queries. 

To search for movies, the app uses `multi-qa-distilbert-cos-v1` model to embed the titles to vector space. Furthermore, for movie recommendation, it uses two distinct embedding techniques: `TF-IDF` (Term Frequency - Inverse Document Frequency) vectors based on the plot (overview) and `Count Vectorization` based on the metadata. 

### Try it out (Demo link)
http://167.172.177.11/

>**NOTE:**
>The data set used for this project, for most movies, doesn't have long plots (overview) and could be vague. This could impact the plot-based recommendations. Similarly, adding more movies to the data could also improve the metadata-based recommendation. Currently, the deployed web-app consists of only `3800` movies as it runs on the free Qdrant cluster with 1GB memory. Due to the memory limit, all the 4802 movie vectors cannot be uploaded even with `on_disk_payload` set to `True`.


## Software requirements

- pip@23.1
- python@3.11.0

## Setup Guide

### Install neural_search Python package

Clone the Neural-Search-with-Qdrant repository and create a virtual environment using `venv` module:
​
```console
foo@bar Neural-Search-with-Qdrant:~$ python3.11 -m venv .venv/
foo@bar Neural-Search-with-Qdrant:~$ source .venv/bin/activate
```

>**NOTE:**
>The virtual environment can be deactivated using the `deactivate` command.
```console
(.venv) foo@bar Neural-Search-with-Qdrant:~$ deactivate
```


Install the neural_search package:
```console
(.venv) foo@bar Neural-Search-with-Qdrant:~$ pip install -e .
```

## Configuring the Movie NeuralBase App

### Connection to Qdrant Cluster
Please make sure that the details to your Qdrant cluster is properly set including the API key. By default, the web-app is configured to run on your local machine. You can either add the `HOST` and `API_KEY` variables to the environment or directly specify it as follows:

```python
#config.py
host = #HOST_NAME
api_key = #API_KEY
```

### Upload Limit
The config file also consits of a `max_data` parameter which determines how many movies to upload in the Qdrant cluster and, consequently, use in the web-app. If you are using the free-tier Qdrant cluster then please set the `max_data` variable to `3800`. Due to the memory limit of 1GB, all the 4802 movie vectors cannot be uploaded even with `on_disk_payload` set to `True`. Similar to the connection details, the max_data can be specified with the "MAX_DATA" in the environment or directly in the config.py:

```python
#config.py
max_data = 3800
```

If the max_data isn't specified then all the data from `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` is read and uploaded to the cluster.

## Deployment Guide

The app can be easily deployed using the scripts under the `demo` directory. By default, the directory environments are setup such that the Python scripts need to be executed from the root directory of the project. However, similar to the previous configurations, it can be modified as desired.

 First, the movie data needs to be embedded and uploaded to the Qdrant cluster in order to be available for the web-app. 
```console
(.venv) foo@bar Neural-Search-with-Qdrant:~$ python demo/populate.py
```

After the vectors are successfully uploaded to the cluster, run the web-app with:
```console
(.venv) foo@bar Neural-Search-with-Qdrant:~$ python demo/app.py
```

The web-app should now be deployed and accessible at http://localhost:8000/.

## Web-App UI

### Home Page

![Home page of the web-app](https://i.imgur.com/d7AqSAn.png "Home Page")

The home page of the web-app is a search bar with random movie suggestions below it. 

### Search Results

![Searching for a movie](https://i.imgur.com/caAvuIb.png "Search Results")

### Movie Page

![Movie Page](https://i.imgur.com/5BQ6NWA.png "Movie Page")

## Contact and Feedback

If you have any feedback, please contact me at prashiddhad.thapa@gmail.com. For any bugs or improvement, I'd be happy to receive your pull request or issues ;)
