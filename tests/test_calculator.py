from app.agent.calculator import calculator


def test_evaluates_arithmetic():
    assert calculator.invoke({"expression": "(1280 - 1056) / 1280 * 100"}).startswith("17.5")


def test_respects_operator_precedence():
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == "14"


def test_rejects_code_execution():
    result = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    assert result.startswith("could not evaluate")


def test_rejects_bare_names():
    assert calculator.invoke({"expression": "x + 1"}).startswith("could not evaluate")


def test_refuses_a_power_tower_without_computing_it():
    # 9**9**9 is 9**387420489, hundreds of millions of digits
    result = calculator.invoke({"expression": "9**9**9"})
    assert result == "could not evaluate '9**9**9': the result would be too large"


def test_refuses_huge_results_built_in_steps():
    assert "too large" in calculator.invoke({"expression": "(10**100)**100"})
    assert "too large" in calculator.invoke({"expression": "(10**1000) * (10**1000) * (10**1000) * (10**1000)"})


def test_ordinary_powers_still_work():
    assert calculator.invoke({"expression": "2**10"}) == "1024"
    assert calculator.invoke({"expression": "1**100000"}) == "1"
    assert calculator.invoke({"expression": "2**-2"}) == "0.25"
    # compound interest: a float base with a big exponent is cheap, so it is allowed
    assert calculator.invoke({"expression": "1000 * (1 + 0.05/365)**(365*30)"}).startswith("4481.")