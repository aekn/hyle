import asyncio
import os
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel

from hyle import Model, Request, Structured, run, tool
from hyle.tools import Tool
from hyle_ollama import OllamaClient, OllamaModel

MODEL = os.getenv("HYLE_MODEL", "qwen3:4b")

SUPPORT_INSTRUCTIONS = (
    "Answer the customer's message. "
    "Use tools for account information; do not guess account details. "
    "Set should_block_card only if the customer reports the card lost, stolen, or compromised, "
    "or asks for it to be blocked."
)


@dataclass(frozen=True, slots=True)
class Transaction:
    merchant: str
    amount: Decimal


@dataclass(frozen=True, slots=True)
class Account:
    name: str
    balance: Decimal
    recent_transactions: tuple[Transaction, ...]


class SupportReply(BaseModel):
    message: str
    should_block_card: bool


SUPPORT_OUTPUT = Structured(SupportReply)


async def answer_customer(
    model: Model,
    account: Account,
    message: str,
) -> SupportReply:
    result = await run(
        model,
        Request(
            _support_prompt(account, message),
            instructions=SUPPORT_INSTRUCTIONS,
            schema=SUPPORT_OUTPUT.schema,
        ),
        tools=_account_tools(account),
    )
    return SUPPORT_OUTPUT(result.response)


async def main() -> None:
    message = "I lost my card. Can you tell me my balance and show me my recent transactions?"
    account = Account(
        name="Eddie Brock",
        balance=Decimal("842.16"),
        recent_transactions=(
            Transaction("Chen's Market", Decimal("58.37")),
            Transaction("Edinburgh Castle Pub", Decimal("18.50")),
            Transaction("STK Steakhouse", Decimal("86.24")),
        ),
    )

    async with OllamaClient() as client:
        model = OllamaModel(MODEL, client=client)
        reply = await answer_customer(model, account, message)

    print(reply.message)
    print(f"block-card: {'yes' if reply.should_block_card else 'no'}")


def _support_prompt(account: Account, message: str) -> str:
    return f"Customer: {account.name}\n\nMessage:\n{message}"


def _account_tools(account: Account) -> tuple[Tool, ...]:
    @tool
    def get_balance() -> Decimal:
        """Return the customer's current account balance."""
        return account.balance

    @tool
    def get_recent_transactions() -> tuple[Transaction, ...]:
        """Return the customer's most recent card transactions."""
        return account.recent_transactions

    return get_balance, get_recent_transactions


if __name__ == "__main__":
    asyncio.run(main())
