from neural_search import NeuralSearch
from neural_search.config import TEMPLATE_DIR

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from urllib.parse import quote_plus, unquote_plus
from fastapi.exceptions import HTTPException

ns = NeuralSearch()

templates = Jinja2Templates(directory=TEMPLATE_DIR)
app = FastAPI()


@app.get("/search/{query}")
def search_movies(query: str, request: Request):

    query = unquote_plus(query)

    movie_info = []

    movie_titles = ns.search_movies(query)

    for movie_title in movie_titles:
        description = ns.get_movie_overview(movie_title)
        genres = ns.get_movie_genres(movie_title)

        movie_info.append(
            {"title": movie_title, "description": description, "genres": genres}
        )

    return templates.TemplateResponse(
        "search.html", {"request": request, "query": query, "suggestions": movie_info}
    )


@app.post("/submit")
def submit(query: str = Form(...)):
    return RedirectResponse(url="/search/" + quote_plus(query, safe="/", encoding='utf8'), status_code=303)


@app.get("/api/similar_plot_movies/")
def search_similar_plot_movies(title: str):
    return {"result": ns.recommend_movies(title, "tfidf")}


@app.get("/api/similar_metadata_movies")
def search_similar_metadata_movies(title: str):
    return {"result": ns.recommend_movies(title, "count")}


@app.get("/movie/{movie_title}")
def movie_page(movie_title: str, request: Request):
    if not ns.movie_exists(movie_title):
        return templates.TemplateResponse("error.html", {"request": request, "error_msg": "we couldn't find any movie with the title: " + movie_title})

    movie_desc = ns.get_movie_overview(movie_title)
    similar_plot_movies = ns.recommend_movies(movie_title, "tfidf")
    similar_metadata_movies = ns.recommend_movies(movie_title, "count")

    if not similar_plot_movies:
        similar_plot_movies = []

    if not similar_metadata_movies:
        similar_metadata_movies = []

    return templates.TemplateResponse(
        "movie.html",
        {
            "request": request,
            "selected_movie": movie_title,
            "description": movie_desc,
            "plot_based_rec": similar_plot_movies,
            "metadata_based_rec": similar_metadata_movies,
        },
    )

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {"request": request, "error_msg": "the following path doesn't exist: " + request.url._url})

@app.get("/")
def homepage(request: Request):
    movie_titles = ns.get_random_movie_titles()
    return templates.TemplateResponse(
        "home.html", {"request": request, "rand_titles": movie_titles}
    )