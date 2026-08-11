from pydantic import BaseModel

class Step(BaseModel):
    id: int
    action: str
    args: dict
    description: str

class Plan(BaseModel):
    goal: str
    steps: list[Step]

