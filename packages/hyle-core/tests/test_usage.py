from hyle.usage import Usage


def test_usage_total_and_addition() -> None:
    assert Usage(2, 3).total == 5
    assert Usage(2, 3) + Usage(5, 7) == Usage(7, 10)


def test_unknown_usage_propagates() -> None:
    assert (Usage(None, 3) + Usage(5, 7)).input is None
    assert Usage(None, 3).total is None
