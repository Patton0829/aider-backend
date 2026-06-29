from fastapi import APIRouter

from app.api.routes import agent, health, job_logs, models


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router, prefix="/api")
api_router.include_router(agent.router)
api_router.include_router(job_logs.router)

