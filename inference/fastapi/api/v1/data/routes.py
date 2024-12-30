from fastapi import APIRouter

from models.responses import AccountIdsListResponse

router = APIRouter()


@router.get(
    "/account_ids",
    response_model=AccountIdsListResponse,
    summary="Получить список всех account_id игроков",
)
async def get_account_ids():
    # TODO: implement
    return AccountIdsListResponse(account_ids=[1, 2, 3, 4, 5])
