from fastapi import APIRouter, status
from .scheme import PCAIn, PCAOut
from PCA import PCA


pca = PCA(2)

modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='PCA fit transform',
    status_code=status.HTTP_200_OK,
    response_model=PCAOut
)
async def pca_fit(
        pca_scheme: PCAIn
) -> PCAOut:
    pca.n_components = 2
    pca_scheme = PCAOut(
        X=pca.fit(pca_scheme.X).tolist()
    )
    return pca_scheme