from pydantic import BaseModel, Json


class TwitterAccount(BaseModel):
    id: str
    username: int
    data: Json

    class Config:
        orm_mode=True
    