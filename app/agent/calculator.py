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


def _evaluate(node):
    # Walk the parsed expression and compute it, allowing only numbers and the operators.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
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