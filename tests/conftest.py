import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


@pytest.fixture
def gradient():
    """Les dérivées partielles centrées d'une fonction scalaire de (x, y)."""

    def calcule(fonction, x, y, h=1e-5):
        return (
            (fonction(x + h, y) - fonction(x - h, y)) / (2 * h),
            (fonction(x, y + h) - fonction(x, y - h)) / (2 * h),
        )

    return calcule
