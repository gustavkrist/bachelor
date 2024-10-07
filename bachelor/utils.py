LEVEL_TO_CEFR = {
    1: "A1",
    2: "A1",
    3: "A1",
    4: "A2",
    5: "A2",
    6: "A2",
    7: "B1",
    8: "B1",
    9: "B1",
    10: "B2",
    11: "B2",
    12: "B2",
    13: "C1",
    14: "C1",
    15: "C1",
    16: "C2",
}


def level_to_cefr(level: int) -> str:
    return LEVEL_TO_CEFR[level]


def fk_to_cefr(score: int | float) -> str:
    # Divided the scale into 6 bands
    if score > 83:
        return "A1"
    elif score > 66:
        return "A2"
    elif score > 49:
        return "B1"
    elif score > 32:
        return "B2"
    elif score > 15:
        return "C1"
    else:
        return "C2"
