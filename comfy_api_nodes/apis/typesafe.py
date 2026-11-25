from typing import Literal

from pydantic import BaseModel, Field


class TypeSafeImageAsset(BaseModel):
    type: Literal["image"] = "image"
    format: Literal["jpg"] = "jpg"
    data_base64: str = Field(...)


class TypeSafeNoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str = Field(...)


class TypeSafeChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str = Field(...)
    criteria: dict[str, str] = Field(...)


class TypeSafeScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str = Field(...)
    criteria: list[str] = Field(...)


class TypeSafeSystemOneRequest(BaseModel):
    model: str = Field(...)
    state: dict[str, str | list[str]] = Field(...)
    questions: dict[str, TypeSafeNoulQuestion | TypeSafeChoiceQuestion | TypeSafeScoreQuestion] = Field(...)
    assets: dict[str, TypeSafeImageAsset] | None = Field(None)


class TypeSafeAnswer(BaseModel):
    type: str = Field(...)
    noul: float | None = Field(None)
    choice: str | None = Field(None)
    score: float | None = Field(None)
    confidence: float | None = Field(None)


class TypeSafeSystemOneResponse(BaseModel):
    model: str = Field(...)
    answers: dict[str, TypeSafeAnswer] = Field(...)
