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
