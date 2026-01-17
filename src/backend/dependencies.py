from fastapi import Request


def get_retriever(request: Request):
    return request.app.state.hybrid_retriever


def get_generator(request: Request):
    return request.app.state.generator


def get_db(request: Request):
    return request.app.state.db


def get_redis(request: Request):
    return request.app.state.cache
