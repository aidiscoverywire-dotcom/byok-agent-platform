"""Domain models."""


class User:
    """A user record."""

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def display(self) -> str:
        return f"{self.name} <{self.email}>"


class Session:
    """A login session."""

    def __init__(self, user: User, token: str) -> None:
        self.user = user
        self.token = token

    def is_valid(self) -> bool:
        return bool(self.token)
