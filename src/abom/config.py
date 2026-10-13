"""Физиологические лимиты, веса влияния органов и стартовое состояние."""
from __future__ import annotations

# Допустимые диапазоны 13 параметров организма.
PHYSIO_LIMITS: dict[str, tuple[float, float]] = {
    "heart_rate":         (30.0, 200.0),
    "stroke_volume":      (20.0, 150.0),
    "contractility":      (0.2, 2.5),
    "blood_pressure":     (40.0, 200.0),
    "blood_volume":       (3.0, 8.0),
    "sodium":             (120.0, 160.0),
    "potassium":          (2.0, 7.0),
    "gfr":                (5.0, 180.0),
    "renin":              (0.0, 10.0),
    "aldosterone":        (0.0, 5.0),
    "oxygen_saturation":  (0.5, 1.0),
    "urine_output":       (0.0, 10.0),
    "cardiac_output":     (1.0, 15.0),
}

# Кто и с каким весом влияет на параметр. Пусто = производный.
AUTHORITY: dict[str, dict[str, float]] = {
    "heart_rate":         {"heart": 1.0},
    "stroke_volume":      {"heart": 1.0},
    "contractility":      {"heart": 1.0},
    "blood_pressure":     {"heart": 0.4, "kidney": 0.6},
    "blood_volume":       {"heart": 0.2, "kidney": 0.8},
    "sodium":             {"kidney": 1.0},
    "potassium":          {"kidney": 1.0},
    "gfr":                {"kidney": 1.0},
    "renin":              {"kidney": 1.0},
    "aldosterone":        {"kidney": 1.0},
    "oxygen_saturation":  {"heart": 0.7, "kidney": 0.3},
    "urine_output":       {"kidney": 1.0},
    "cardiac_output":     {},  # производный
}

# Стартовое состояние: лёгкая гиповолемия, чуть ниже нормы сатурация.
INITIAL_BODY: dict[str, float] = {
    "heart_rate": 60.0,
    "stroke_volume": 70.0,
    "contractility": 1.0,
    "blood_pressure": 120.0,
    "blood_volume": 4.5,
    "sodium": 145.0,
    "potassium": 4.5,
    "gfr": 90.0,
    "renin": 1.0,
    "aldosterone": 1.0,
    "oxygen_saturation": 0.94,
    "urine_output": 1.5,
    "cardiac_output": 4.2,
}

# Параметры прогона.
MAX_ITERATIONS = 8
RECURSION_LIMIT = 100
