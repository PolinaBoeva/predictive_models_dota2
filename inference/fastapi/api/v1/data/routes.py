from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/account_ids",
    summary="Получить список всех account_id игроков",
)
async def get_account_ids():
    return {"account_ids": [1, 2, 3, 4, 5]}
