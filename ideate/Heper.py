from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Validated transaction data."""

    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    currency: str = Field(default="USD", min_length=3, max_length=3)


class PaymentGateway:
    total_tx = 0

    def __init__(self, name: str):
        self.name = name

    def process(self, amt: float | Transaction):
        """Process a payment. Accepts amount (float) or a validated Transaction."""
        if isinstance(amt, Transaction):
            amount = amt.amount
            print(self.name, amount, amt.currency)
        else:
            print(self.name, amt)
        PaymentGateway.total_tx += 1

    @classmethod
    def stats(cls) -> int:
        return cls.total_tx

    @staticmethod
    def validate(amount: float) -> bool:
        """Validate amount using Pydantic (must be > 0)."""
        try:
            Transaction(amount=amount)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    pg = PaymentGateway("abc")
    pg.process(10)
    pg.process(Transaction(amount=25.50, currency="USD"))
    print("Stats:", pg.stats())
    print("Valid 100:", pg.validate(100))
    print("Valid -1:", pg.validate(-1))
