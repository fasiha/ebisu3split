from scipy.optimize import minimize_scalar  # type:ignore
from math import log2
from typing import Optional, Tuple
import ebisu as ebisu2

HOURS_PER_YEAR = 365 * 24
Ebisu2Model = Tuple[float, float, float]
SubModel = Tuple[float, float, float, float]
Model = Tuple[SubModel, SubModel, SubModel]


def predictRecallBetaPowerlaw(model, elapsed):
    delta = elapsed / model[-1]
    l = log2(1 + delta)
    return ebisu2.predictRecall(model, l * model[-1], exact=True)


def norm(v: list[float]) -> list[float]:
    s = sum(v)
    return [x / s for x in v]


def initModel(alphaBeta: float,
              hlHours: float,
              w1=0.65,
              w2=0.3,
              scale2=2) -> Model:
    ws = norm([w1, w2, 1 - w1 - w2])
    return (
        (ws[0], alphaBeta, alphaBeta, hlHours),
        (ws[1], alphaBeta, alphaBeta, hlHours * scale2),
        (ws[2], alphaBeta, alphaBeta, HOURS_PER_YEAR),
    )


def predictRecall(model: Model, elapsed: float) -> float:
    primary, strength, longterm = model
    return (
        primary[0] * ebisu2.predictRecall(primary[1:], elapsed, exact=True) +
        strength[0] * ebisu2.predictRecall(strength[1:], elapsed, exact=True) +
        longterm[0] * predictRecallBetaPowerlaw(longterm[1:], elapsed))


def updateRecall(model: Model,
                 successes: float,
                 total: int,
                 elapsed: float,
                 q0: Optional[float] = None) -> Model:
    scale2 = model[1][-1] / model[0][-1]
    newPrimary: Ebisu2Model = ebisu2.updateRecall(model[0][1:],
                                                  successes,
                                                  total,
                                                  elapsed,
                                                  q0=q0)
    strength: SubModel = (*model[1][:-1], newPrimary[-1] * scale2)
    return ((model[0][0], *newPrimary), strength, model[2])


def modelToPercentileDecay(model: Model, percentile=0.5) -> float:
    logLeft, logRight = 0, 0
    counter = 0
    while predictRecall(model, 10**logLeft) <= percentile:
        logLeft -= 1
        counter += 1
        if counter >= 20:
            raise Exception('unable to find left bound')

    counter = 0
    while predictRecall(model, 10**logRight) >= percentile:
        logRight += 1
        counter += 1
        if counter >= 20:
            raise Exception('unable to find right bound')

    res = minimize_scalar(lambda h: abs(percentile - predictRecall(model, h)),
                          bounds=[10**logLeft, 10**logRight])
    assert res.success
    return res.x
