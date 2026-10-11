from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Importance = Literal["Required", "Preferred"]
EvaluationStatus = Literal["Supported", "Unverified", "Missing"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


# --- API input schemas ---
class JobDescriptionInput(StrictBaseModel):
    raw_text: str = Field(min_length=30, max_length=30_000)


# --- Structured extraction schemas ---
class JobRequirement(StrictBaseModel):
    skill_name: str = Field(min_length=1, max_length=500)
    # Nullable: the extractor prompt may legitimately not know the importance.
    # The API layer defaults None -> "Preferred" after validation.
    importance: Importance | None = None


class ExtractedJobDescription(StrictBaseModel):
    role: str = Field(min_length=1, max_length=200)
    requirements: list[JobRequirement] = Field(min_length=1, max_length=100)
    minimum_experience_years: float | None = Field(
        default=None,
        ge=0,
        le=80,
    )


class Experience(StrictBaseModel):
    company: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=200)
    duration: str | None = Field(default=None, max_length=100)
    skills_used: list[str] = Field(default_factory=list, max_length=100)


class ExtractedResume(StrictBaseModel):
    # Nullable: if the resume does not state a name, the API layer substitutes
    # "Unknown Candidate". A non-nullable default would silently mislabel.
    name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=50)
    skills: list[str] = Field(default_factory=list, max_length=200)
    experiences: list[Experience] = Field(default_factory=list, max_length=100)

    # Intentionally no "meaningful_chunks" field.
    # Evidence chunks are built from the original PDF text in the API layer.


# --- Requirement evaluation and API response schemas ---
class RequirementEvaluation(StrictBaseModel):
    # 500 chars so long JD bullets (e.g., Amazon "Cloud...and/or VCS") survive.
    requirement: str = Field(min_length=1, max_length=500)
    # Nullable: the scorer assigns this from the caller's input after parsing.
    importance: Importance | None = None
    status: EvaluationStatus
    evidence_quote: str | None = Field(default=None, max_length=2000)
    reasoning: str = Field(min_length=1, max_length=2000)

    @field_validator("evidence_quote")
    @classmethod
    def empty_quote_becomes_none(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class MatchResult(StrictBaseModel):
    candidate_name: str
    overall_score: int = Field(ge=0, le=100)
    evaluations: list[RequirementEvaluation]

    # True only when every requirement was evaluated successfully.
    evaluation_complete: bool

    # Warnings are deliberately generic; never include resume text or PII.
    warnings: list[str] = Field(default_factory=list)

    score_explanation: str = (
        "Weighted requirement coverage: Supported Required requirements "
        "count as 2 points, Supported Preferred requirements count as 1 "
        "point. The score is not a hiring probability."
    )
