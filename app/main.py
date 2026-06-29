from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.services.aider import AiderSessionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.aider_sessions = AiderSessionManager()
    yield
    await app.state.aider_sessions.close_all()


settings = get_settings()
app = FastAPI(
    title="Control Center Aider Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)

