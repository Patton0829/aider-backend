from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from app.code_conversion.service import (
    ConversionJobConflictError,
    ConversionJobNotFoundError,
    ConversionUploadError,
)
from app.core.config import get_settings


router = APIRouter(prefix="/api/code-conversion", tags=["code-conversion"])


def authorize(request: Request) -> None:
    expected = get_settings().agent_api_token
    if not expected:
        return
    authorization = request.headers.get("authorization", "")
    supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/jobs/{job_id}", status_code=status.HTTP_202_ACCEPTED)
async def submit_conversion(
    job_id: str,
    request: Request,
    filename: str = Query(default="algorithm.zip", min_length=1, max_length=255),
):
    authorize(request)
    try:
        job = await request.app.state.code_conversion_service.submit(
            job_id, filename, request.stream()
        )
    except ConversionJobConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ConversionUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return job.public_dict()


@router.get("/jobs/{job_id}")
async def conversion_status(job_id: str, request: Request):
    authorize(request)
    try:
        job = await request.app.state.code_conversion_service.get(job_id)
    except (ConversionJobNotFoundError, ConversionUploadError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return job.public_dict()


@router.get("/jobs/{job_id}/artifact")
async def conversion_artifact(job_id: str, request: Request):
    authorize(request)
    try:
        artifact = await request.app.state.code_conversion_service.artifact(job_id)
    except (ConversionJobNotFoundError, ConversionUploadError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(
        artifact,
        media_type="application/zip",
        filename=f"{job_id}_python.zip",
    )


@router.get("/jobs/{job_id}/files")
async def conversion_files(job_id: str, request: Request):
    authorize(request)
    try:
        files = await request.app.state.code_conversion_service.converted_files(job_id)
    except (ConversionJobNotFoundError, ConversionUploadError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [file.public_dict() for file in files]


@router.get("/jobs/{job_id}/files/content")
async def conversion_file_content(
    job_id: str,
    request: Request,
    path: str = Query(min_length=1, max_length=1024),
):
    authorize(request)
    try:
        file = await request.app.state.code_conversion_service.converted_file_content(
            job_id, path
        )
    except ConversionJobNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ConversionUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return file.public_dict() | {"content": file.content}


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversion(job_id: str, request: Request):
    authorize(request)
    try:
        await request.app.state.code_conversion_service.remove(job_id)
    except ConversionJobNotFoundError:
        return None
    except ConversionUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return None
