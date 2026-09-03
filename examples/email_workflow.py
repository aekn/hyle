import asyncio
import os
from dataclasses import dataclass

from pydantic import BaseModel

from hyle import Model, Request, Structured
from hyle_ollama import OllamaClient, OllamaGenerationConfig, OllamaModel

MODEL = os.getenv("HYLE_MODEL", "qwen3:4b")

DRAFT_INSTRUCTIONS = (
    "Draft the email from the brief. "
    "Use only information in the brief. "
    "Keep it concise and natural."
)

REVIEW_INSTRUCTIONS = (
    "Review the draft against the brief. "
    "Check factual accuracy, clarity, and tone. "
    "Approve it only if it fully satisfies the brief and adds no unsupported details. "
    "If it is not approved, list the changes needed."
)


class Email(BaseModel):
    subject: str
    body: str


class EmailReview(BaseModel):
    approved: bool
    issues: tuple[str, ...]


EMAIL_OUTPUT = Structured(Email)
REVIEW_OUTPUT = Structured(EmailReview)


@dataclass(frozen=True, slots=True)
class EmailWriter:
    model: Model
    max_revisions: int = 2

    async def run(self, brief: str, /) -> Email:
        draft = await self._draft(brief)

        for revision in range(self.max_revisions + 1):
            review = await self._review(brief, draft)
            if review.approved:
                return draft
            if revision == self.max_revisions:
                break
            draft = await self._revise(brief, draft, review.issues)

        raise RuntimeError("email did not pass review")

    async def _draft(self, brief: str) -> Email:
        response = await self.model.generate(
            Request(
                brief,
                instructions=DRAFT_INSTRUCTIONS,
                schema=EMAIL_OUTPUT.schema,
            )
        )
        return EMAIL_OUTPUT(response)

    async def _revise(self, brief: str, draft: Email, issues: tuple[str, ...]) -> Email:
        response = await self.model.generate(
            Request(
                _revision_prompt(brief, draft, issues),
                instructions=DRAFT_INSTRUCTIONS,
                schema=EMAIL_OUTPUT.schema,
            )
        )
        return EMAIL_OUTPUT(response)

    async def _review(self, brief: str, draft: Email) -> EmailReview:
        response = await self.model.generate(
            Request(
                _review_prompt(brief, draft),
                instructions=REVIEW_INSTRUCTIONS,
                schema=REVIEW_OUTPUT.schema,
            )
        )
        return REVIEW_OUTPUT(response)


async def main() -> None:
    brief = (
        "Email the team that tomorrow's launch is moving to Monday because final testing "
        "is not complete. Thank them for the extra work and keep the tone calm and direct."
    )

    async with OllamaClient() as client:
        model = OllamaModel(
            MODEL,
            client=client,
            config=OllamaGenerationConfig(think=False),
        )
        email = await EmailWriter(model).run(brief)

    print(f"[{email.subject}]\n")
    print(email.body)


def _revision_prompt(brief: str, draft: Email, issues: tuple[str, ...]) -> str:
    return (
        f"Brief:\n{brief}\n\n"
        f"Draft:\n"
        f"Subject: {draft.subject}\n\n"
        f"{draft.body}\n\n"
        f"Issues:\n"
        + "\n".join(f"- {issue}" for issue in issues)
    )  # fmt: skip


def _review_prompt(brief: str, draft: Email) -> str:
    return (
        f"Brief:\n{brief}\n\n"
        f"Draft:\n"
        f"Subject: {draft.subject}\n\n"
        f"{draft.body}"
    )  # fmt: skip


if __name__ == "__main__":
    asyncio.run(main())
