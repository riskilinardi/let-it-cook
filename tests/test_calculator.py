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