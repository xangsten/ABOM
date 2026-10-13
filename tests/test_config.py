"""Инварианты конфигурации организма."""
from abom.config import AUTHORITY, INITIAL_BODY, PHYSIO_LIMITS


def test_authority_covers_exactly_all_params():
    assert set(AUTHORITY) == set(PHYSIO_LIMITS)


def test_initial_body_within_limits():
    assert set(INITIAL_BODY) == set(PHYSIO_LIMITS)
    for param, value in INITIAL_BODY.items():
        lo, hi = PHYSIO_LIMITS[param]
        assert lo <= value <= hi, f"{param}={value} вне диапазона [{lo}, {hi}]"


def test_cardiac_output_is_derived():
    # Пустой AUTHORITY = параметр производный, органы им не управляют.
    assert AUTHORITY["cardiac_output"] == {}


def test_authority_weights_are_positive():
    for param, weights in AUTHORITY.items():
        for organ, w in weights.items():
            assert w > 0, f"нулевой вес: {param}/{organ}"
