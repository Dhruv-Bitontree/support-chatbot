"""Custom exception classes."""


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class OrderNotFoundError(AppError):
    def __init__(self, order_id: str):
        super().__init__(f"Order '{order_id}' not found", status_code=404)


class TicketNotFoundError(AppError):
    def __init__(self, ticket_id: str):
        super().__init__(f"Ticket '{ticket_id}' not found", status_code=404)


class LLMProviderError(AppError):
    def __init__(self, provider: str, detail: str):
        super().__init__(f"LLM provider '{provider}' error: {detail}", status_code=502)


class VectorStoreError(AppError):
    def __init__(self, detail: str):
        super().__init__(f"Vector store error: {detail}", status_code=500)


class RateLimitError(AppError):
    def __init__(self):
        super().__init__("Rate limit exceeded. Please try again later.", status_code=429)
