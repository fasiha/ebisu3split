from scipy.optimize import minimize_scalar  # type:ignore
from math import log2
from typing import Optional, Tuple
import ebisu as ebisu2  # type:ignore

HOURS_PER_YEAR = 365 * 24
Ebisu2Model = Tuple[float, float, float]
SubModel = Tuple[float, float, float, float]
Model = Tuple[SubModel, SubModel, SubModel]


def predictRecallBetaPowerlaw(model: Ebisu2Model, elapsed: float):
    """Like predictRecall but assumes recall decays by power-law

    (Instead of exponential.) See https://github.com/fasiha/ebisu/issues/64
    """
    delta = elapsed / model[-1]
    l = log2(1 + delta)
    return ebisu2.predictRecall(model, l * model[-1], exact=True)


def norm(v: list[float]) -> list[float]:
    "Normalizes a list to unit-norm by dividing elements by its sum"
    s = sum(v)
    return [x / s for x in v]


def initModel(alphaBeta: float,
              hl1: float,
              w1=0.35,
              w2=0.35,
              scale2=5,
              hl3=HOURS_PER_YEAR * 10) -> Model:
    """Initializes an Ebisu 3split model

    This consists of three models:

    First, a primary Ebisu2 model (exponentially-decaying memory).

    Second, a secondary "strengthening" Ebisu2 model (again, exponential decay), which is
    never updated. That is, its α, β parameters are fixed, but its time parameter
    (half-life) is a constant multiple of the first model's time parameter. This is
    `scale2`.

    The third and final model is a long-term power-law model, where memory decays
    according to a power-law, rather than an exponential. This model is also never
    updated, neither its α and β parameters nor its time parameter.

    `alphaBeta` governs all three models' α and β parameters.

    `hl1` is the primary model's halflife. You are free to use whatever units you want but
    do note that the default for `hl3` (below) assumes this argument is in hours.

    `w1` and `w2` are floats between 0 and 1 that govern the weight of the first and
    second models. The third model's weight will be treated as `1 - w1 - w2`.

    `scale2` as mentioned above specifies the second model's constant multiple offset of
    the primary model's halflife. If `scale2=5` and you pass in `hl1=1` hour, the second
    model's halflife will be 5 hours.

    `hl3` is the third (long-term) model's half-life. Since this is never updated, it
    should be some kind of upper-bound on what time horizon effectively means "permanently
    learned". By default this is the number of hours in ten years.
    """
    ws = norm([w1, w2, 1 - w1 - w2])
    return (
        (ws[0], alphaBeta, alphaBeta, hl1),
        (ws[1], alphaBeta, alphaBeta, hl1 * scale2),
        (ws[2], alphaBeta, alphaBeta, hl3),
    )


def predictRecall(model: Model, elapsed: float) -> float:
    """Predict the current recall of an Ebisu 3split model

    Assumes `elapsed` is in units consistent with the model, and is the time elapsed since
    the model was created (either by `initModel` or by `updateModel`).

    Returns a probability between 0 and 1. The higher the number, the more likely you are
    to remember this fact.
    """
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
    """Update an Ebisu 3split model with a new quiz

    The new quiz is either a binomial quiz, `successes` passes out of a `total` number of
    trials (e.g., Duolingo data, in which case `0 <= successes <= total`), or fuzzy-binary
    quiz where `total=1` and `0 <= successes <= 1` can be a float.

    `elapsed` is the time elapsed since the model was created (either by `initModel` or
    `updateModel` itself), in units consistent with the model.

    `q0` applies only to the fuzzy-binary case and is explained in detail in the Ebisu
    documentation https://fasiha.github.io/ebisu/ but in a nutshell is the probability
    that you passed the quiz (`successes > 0.5`) given that you actually forgot the fact.
    By default it'll be `1 - max(successes, 1 - successes)`.
    """
    scale2 = model[1][-1] / model[0][-1]
    newPrimary: Ebisu2Model = ebisu2.updateRecall(model[0][1:],
                                                  successes,
                                                  total,
                                                  elapsed,
                                                  q0=q0)
    strength: SubModel = (*model[1][:-1], newPrimary[-1] * scale2)
    return ((model[0][0], *newPrimary), strength, model[2])


def modelToPercentileDecay(model: Model, percentile=0.5) -> float:
    """How long does it take for an Ebisu 3split model to decay to a given percent recall?

    `percentile` is between 0 and 1. The default, 0.5, will return the half-life (time for
    the model to decay to 50%).
    """
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
