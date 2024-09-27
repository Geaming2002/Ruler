from collections import OrderedDict


class Inf:
    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __lt__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, Inf)

    def __repr__(self):
        return "Inf"


inf = Inf()


# FLCG EXP
LEVEL0 = ["10", "30", "50", "80"]
LEVEL1 = ["150", "300", "500"]
LEVEL2 = ["700", ">800"]
RANGE = OrderedDict(
    {
        # level:0
        "10": {"PM": [0, 20], "FM": [0, 20]},
        "30": {"PM": [20, 40], "FM": [20, 40]},
        "50": {"PM": [40, 60], "FM": [40, 60]},
        "80": {"PM": [70, 90], "FM": [60, 100]},
        # level:1
        "150": {"PM": [130, 170], "FM": [100, 200]},
        "300": {"PM": [280, 320], "FM": [200, 400]},
        "500": {"PM": [450, 550], "FM": [400, 600]},
        # level:2
        "700": {"PM": [630, 770], "FM": [600, 800]},
        ">800": {"PM": [800, inf], "FM": [800, inf]},
    }
)

TARGET_LENGTH = list(RANGE.keys())

MetaLengthToken = [
    ["[MLT:10]", [5, 15]],
    ["[MLT:30]", [25, 35]],
    ["[MLT:50]", [45, 55]],
    ["[MLT:80]", [75, 85]],
    ["[MLT:150]", [135, 155]],
    ["[MLT:300]", [295, 305]],
    ["[MLT:500]", [495, 505]],
    ["[MLT:700]", [695, 705]],
    ["[MLT:>800]", [800, inf]],
]

# MLT training dataset
SAMPLE = {
    "[MLT:10]": 10000 * 2,
    "[MLT:30]": 10000 * 2,
    "[MLT:50]": 10000 * 2,
    "[MLT:80]": 10000 * 2,
    "[MLT:150]": 10000 * 2,
    "[MLT:300]": 10000 * 2,
    "[MLT:500]": 10000 * 2,
    "[MLT:700]": 10000 * 2,
    "[MLT:>800]": 10000 * 2,
}
