import ast
import operator

from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Python integers have no size limit, so something like 9**9**9 would tie up the
# worker computing a number with hundreds of millions of digits. Cap integer
# results at 10,000 bits (about 3,000 digits), far beyond any real calculation.
MAX_RESULT_BITS = 10_000


def _check_power(base, exponent):
    # Only an integer raised to a positive integer power can grow without bound;
    # float powers are fast and overflow with an error on their own. The result of
    # base**exponent has at most bit_length(base) * exponent bits, so refuse it
    # before computing anything.
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1:
        if abs(base).bit_length() * exponent > MAX_RESULT_BITS:
            raise ValueError("the result would be too large")


def _check_size(value):
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise ValueError("the result would be too large")
    return value


def _evaluate(node):
    # Walk the parsed expression and compute it, allowing only numbers and the operators.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _check_size(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _check_size(_ALLOWED_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only plain arithmetic is allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.
    Use this for any calculation instead of working it out yourself, so the numbers
    are right. Supports + - * / // % ** and parentheses, for example
    "(1280 - 1056) / 1280 * 100".
    """
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"could not evaluate '{expression}': {exc}"