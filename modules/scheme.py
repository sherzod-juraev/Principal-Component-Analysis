from pydantic import BaseModel, field_validator
from numpy import array, isnan, where, nanmean, take, nan
from fastapi import HTTPException, status


class PCAOut(BaseModel):

    X: list[list]

class PCAIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list]

    @field_validator('X')
    def verify_X(cls, value):
        X = array([[nan if v is None else v for v in row] for row in value])
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='X must be 2D matrix'
            )
        col_mean = nanmean(X, axis=0)
        inds = where(isnan(X))
        X[inds] = take(col_mean, inds[1])
        return X