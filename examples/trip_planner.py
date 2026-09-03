import asyncio
import os
from dataclasses import dataclass

from hyle import Model, Request
from hyle_ollama import OllamaClient, OllamaGenerationConfig, OllamaModel

MODEL = os.getenv("HYLE_MODEL", "qwen3:4b")

PLANNER_INSTRUCTIONS = (
    "Produce a clear, coherent itinerary from the specialist notes. "
    "Resolve conflicts rather than repeating alternatives. "
    "Do not invent current prices, opening hours, availability, or reservations."
)

LOCAL_GUIDE_INSTRUCTIONS = (
    "Suggest a small number of places and neighborhoods that fit the trip. "
    "Briefly explain why each is worth including. "
    "Do not invent current prices or opening hours."
)

LOGISTICS_INSTRUCTIONS = (
    "Focus on pace and logistics. "
    "Suggest a sensible daily rhythm, group nearby activities, and leave room for rest "
    "and spontaneity."
)


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    model: Model
    instructions: str

    async def run(self, task: str, /, *, notes: str | None = None) -> str:
        prompt = task if notes is None else _planning_prompt(task, notes)
        response = await self.model.generate(Request(prompt, instructions=self.instructions))
        return response.require_text()


@dataclass(frozen=True, slots=True)
class Crew:
    planner: Agent
    specialists: tuple[Agent, ...]

    async def run(self, task: str, /) -> str:
        async with asyncio.TaskGroup() as group:
            jobs = [
                (agent, group.create_task(agent.run(task), name=agent.name))
                for agent in self.specialists
            ]

        notes = "\n\n".join(
            f"[{agent.name}]\n{job.result()}"
            for agent, job in jobs
        )  # fmt: skip
        return await self.planner.run(task, notes=notes)


async def main() -> None:
    task = (
        "Plan a relaxed three-day trip to Kyoto for two friends who enjoy food, parks, "
        "and museums. Keep the plan practical and leave some free time each day."
    )

    async with OllamaClient() as client:
        model = OllamaModel(
            MODEL,
            client=client,
            config=OllamaGenerationConfig(think=False),
        )
        crew = Crew(
            planner=Agent("planner", model, PLANNER_INSTRUCTIONS),
            specialists=(
                Agent("local-guide", model, LOCAL_GUIDE_INSTRUCTIONS),
                Agent("logistics", model, LOGISTICS_INSTRUCTIONS),
            ),
        )
        plan = await crew.run(task)

    print(plan)


def _planning_prompt(task: str, notes: str) -> str:
    return f"{task}\n\nSpecialist notes:\n{notes}"


if __name__ == "__main__":
    asyncio.run(main())
