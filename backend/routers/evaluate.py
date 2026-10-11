import logging
import os
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from models.schemas import (
    Importance,
    MatchResult,
    RequirementEvaluation,
)
from services.extractor import extract_job_data, extract_resume_data
from services.scorer import (
    chunk_resume_text,
    get_embeddings,
    retrieve_top_chunks,
    verify_requirement,
)


logger = logging.getLogger(__name__)
router = APIRouter()

MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
MAX_EXTRACTED_CHARS = int(os.getenv("MAX_EXTRACTED_CHARS", "100000"))
MAX_REQUIREMENTS = int(os.getenv("MAX_REQUIREMENTS", "100"))


def _extract_pdf_text(content: bytes) -> str:
    """Read text from a PDF while rejecting common invalid inputs."""

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

    # MIME types can be unreliable, so validate the actual file signature.
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported.",
            )

        if len(reader.pages) == 0:
            raise HTTPException(
                status_code=422,
                detail="The PDF contains no pages.",
            )

        if len(reader.pages) > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"The PDF exceeds the {MAX_PDF_PAGES}-page limit.",
            )

        page_texts: list[str] = []

        for page in reader.pages:
            # Some PDFs have pages where extract_text() returns None.
            page_texts.append(page.extract_text() or "")

        raw_text = "\n\n".join(page_texts).strip()

    except HTTPException:
        raise
    except (PdfReadError, ValueError, OSError) as exc:
        logger.warning("Could not read uploaded PDF.")
        raise HTTPException(
            status_code=400,
            detail="The PDF is malformed or could not be read.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while reading a PDF.")
        raise HTTPException(
            status_code=400,
            detail="The PDF could not be processed.",
        ) from exc

    if not raw_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No selectable text was found in this PDF. "
                "It may be a scanned document; upload a text-based PDF."
            ),
        )

    if len(raw_text) > MAX_EXTRACTED_CHARS:
        raise HTTPException(
            status_code=413,
            detail="The extracted document text exceeds the allowed limit.",
        )

    return raw_text


def _make_failed_evaluation(
    requirement: str,
    importance: Importance,
) -> RequirementEvaluation:
    """Represent a failed verification explicitly instead of skipping it."""

    return RequirementEvaluation(
        requirement=requirement,
        importance=importance,
        status="Unverified",
        evidence_quote=None,
        reasoning=(
            "This requirement could not be verified because the verification "
            "service failed. It was not counted as supported."
        ),
    )


def _calculate_weighted_score(
    evaluations: list[RequirementEvaluation],
) -> int:
    """
    Calculate weighted requirement coverage.

    Required = 2 points; Preferred = 1 point.
    Only Supported evaluations earn points.
    """

    if not evaluations:
        return 0

    total_weight = 0
    supported_weight = 0

    for evaluation in evaluations:
        weight = 2 if evaluation.importance == "Required" else 1
        total_weight += weight

        if evaluation.status == "Supported":
            supported_weight += weight

    if total_weight == 0:
        return 0

    return round((supported_weight / total_weight) * 100)


@router.post("/evaluate", response_model=MatchResult)
async def evaluate_candidate(
    job_description: str = Form(..., min_length=30, max_length=30_000),
    resume_pdf: UploadFile = File(...),
):
    # 1. Validate basic input.
    job_description = job_description.strip()

    if len(job_description) < 30:
        raise HTTPException(
            status_code=422,
            detail="Please provide a more complete job description.",
        )

    # 2. Read the PDF with a strict size limit.
    try:
        content = await resume_pdf.read(MAX_PDF_BYTES + 1)
    finally:
        await resume_pdf.close()

    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB "
                "upload limit."
            ),
        )

    raw_resume_text = _extract_pdf_text(content)

    # 3. Extract structured information. Evidence will still come from the
    # original PDF text, not from the LLM's extracted summary.
    try:
        jd_data = await extract_job_data(job_description)
        resume_data = await extract_resume_data(raw_resume_text)
    except Exception as exc:
        logger.warning("Document extraction failed.")
        raise HTTPException(
            status_code=502,
            detail=(
                "The extraction service could not process the documents. "
                "Please retry shortly."
            ),
        ) from exc

    requirements = jd_data.requirements

    if not requirements:
        raise HTTPException(
            status_code=422,
            detail="No evaluable requirements were extracted from the job description.",
        )

    if len(requirements) > MAX_REQUIREMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"The job description has too many requirements (limit {MAX_REQUIREMENTS}).",
        )

    # 4. Build evidence chunks directly from the original PDF text.
    chunks = chunk_resume_text(raw_resume_text)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No usable text could be extracted from the resume.",
        )

    # 5. Embed all resume chunks in one batch.
    # Document passages are embedded without the BGE query instruction.
    try:
        chunk_vectors = await get_embeddings(chunks, is_query=False)

        # All requirements are query-like inputs, so embed them together
        # with the recommended BGE query instruction.
        requirement_texts = [req.skill_name for req in requirements]
        requirement_vectors = await get_embeddings(
            requirement_texts,
            is_query=True,
        )
    except Exception as exc:
        logger.exception("Embedding generation failed.")
        raise HTTPException(
            status_code=503,
            detail="The embedding service is temporarily unavailable.",
        ) from exc

    if len(chunk_vectors) != len(chunks):
        logger.error("Embedding output count did not match resume chunks.")
        raise HTTPException(
            status_code=503,
            detail="Could not generate reliable resume embeddings.",
        )

    if len(requirement_vectors) != len(requirements):
        logger.error("Embedding output count did not match requirements.")
        raise HTTPException(
            status_code=503,
            detail="Could not generate reliable requirement embeddings.",
        )

    # 6. Retrieve original resume evidence and verify each requirement.
    evaluations: list[RequirementEvaluation] = []
    verification_failed = False

    for index, req in enumerate(requirements):
        requirement_text = req.skill_name
        importance = req.importance

        top_chunks = retrieve_top_chunks(
            requirement_vector=requirement_vectors[index],
            chunk_vectors=chunk_vectors,
            chunks=chunks,
            top_k=3,
        )

        try:
            evaluation = await verify_requirement(
                requirement=requirement_text,
                top_chunks=top_chunks,
                importance=importance,
            )
            evaluations.append(evaluation)

        except Exception:
            # Keep the requirement in the denominator. Never silently skip it.
            logger.warning(
                "Verification failed for a requirement; marking Unverified."
            )
            evaluations.append(
                _make_failed_evaluation(requirement_text, importance)
            )
            verification_failed = True

    # 7. Compute a transparent weighted coverage score.
    overall_score = _calculate_weighted_score(evaluations)

    warnings: list[str] = []
    if verification_failed:
        warnings.append(
            "One or more requirements could not be verified because of a "
            "service error. The score is provisional; retry the evaluation."
        )

    # 8. Return the report.
    return MatchResult(
        candidate_name=resume_data.name or "Unknown Candidate",
        overall_score=overall_score,
        evaluations=evaluations,
        evaluation_complete=not verification_failed,
        warnings=warnings,
    )

