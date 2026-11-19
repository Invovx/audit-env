"""safe_calc.py - a small, safe arithmetic evaluator.

Built on ``ast`` + ``decimal.Decimal``.  There is no ``eval``/``exec``/``compile``
of user code: the text is parsed to an AST, a strict character whitelist and a
node whitelist are applied, and the tree is walked with Decimal arithmetic.

Supported: numbers (``12``, ``485.2500``, ``1e3``), ``+ - * / // % **``,
unary ``+``/``-`` and parentheses.  Everything else (names, calls, attributes,
strings, comparisons, ...) is rejected.
"""
from __future__ import annotations

import ast
import decimal
import re
from decimal import Decimal

MAX_EXPRESSION_CHARS = 200
MAX_AST_NODES = 80
MAX_DEPTH = 64
MAX_ABS_EXPONENT = 100
MAX_ADJUSTED = 60  # |value| must stay below 10**61 (and above ~10**-61 when non-zero)


class CalcError(ValueError):
    """The expression is invalid or cannot be evaluated."""


class UnsafeExpressionError(CalcError):
    """The expression uses a construct that is not allowed."""


_ALLOWED_CHARS = re.compile(r"[0-9+\-*/%().eE_ \t]+")


def _new_context() -> decimal.Context:
    return decimal.Context(
        prec=50,
        rounding=decimal.ROUND_HALF_UP,
        Emin=-999,
        Emax=999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
    )


def _checked(value: Decimal, ctx: decimal.Context) -> Decimal:
    if value.is_nan() or value.is_infinite():
        raise CalcError("result is not a finite number")
    if value != 0 and abs(value.adjusted()) > MAX_ADJUSTED:
        raise CalcError("number is outside the supported range")
    return ctx.plus(value)


def _eval(node: ast.AST, source: str, ctx: decimal.Context, depth: int = 0) -> Decimal:
    if depth > MAX_DEPTH:
        raise CalcError("expression is nested too deeply")

    if isinstance(node, ast.Expression):
        return _eval(node.body, source, ctx, depth + 1)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsafeExpressionError("only numeric literals are allowed")
        try:
            if isinstance(value, int):
                number = Decimal(value)
            else:
                # Use the original literal text: parsing it as float would lose digits.
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    raise CalcError("cannot read numeric literal")
                number = Decimal(segment)
        except decimal.DecimalException:
            raise CalcError("invalid numeric literal") from None
        return _checked(number, ctx)

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, source, ctx, depth + 1)
        if isinstance(node.op, ast.UAdd):
            return _checked(ctx.plus(operand), ctx)
        if isinstance(node.op, ast.USub):
            return _checked(ctx.minus(operand), ctx)
        raise UnsafeExpressionError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval(node.left, source, ctx, depth + 1)
        right = _eval(node.right, source, ctx, depth + 1)
        op = type(node.op)
        if op not in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow):
            raise UnsafeExpressionError("unsupported operator")
        try:
            if op is ast.Add:
                result = ctx.add(left, right)
            elif op is ast.Sub:
                result = ctx.subtract(left, right)
            elif op is ast.Mult:
                result = ctx.multiply(left, right)
            elif op is ast.Div:
                result = ctx.divide(left, right)
            elif op is ast.FloorDiv:
                result = ctx.divide_int(left, right)
            elif op is ast.Mod:
                result = ctx.remainder(left, right)
            else:
                if right != right.to_integral_value() or abs(right) > MAX_ABS_EXPONENT:
                    raise CalcError("exponent must be an integer with absolute value <= 100")
                result = ctx.power(left, right)
        except decimal.DivisionByZero:
            raise CalcError("division by zero") from None
        except decimal.InvalidOperation:
            if right == 0 and op in (ast.Div, ast.FloorDiv, ast.Mod):
                raise CalcError("division by zero") from None
            raise CalcError("invalid arithmetic operation") from None
        except decimal.Overflow:
            raise CalcError("result is too large") from None
        return _checked(result, ctx)

    raise UnsafeExpressionError("unsupported expression element")


def safe_eval(expression: str) -> Decimal:
    """Evaluate ``expression`` and return an exact-as-possible ``Decimal``."""
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    text = expression.strip()
    if not text:
        raise CalcError("expression is empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalcError(f"expression is longer than {MAX_EXPRESSION_CHARS} characters")
    if _ALLOWED_CHARS.fullmatch(text) is None:
        raise UnsafeExpressionError(
            "expression contains unsupported characters; "
            "only numbers and + - * / // % ** ( ) are allowed")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise CalcError("invalid arithmetic syntax") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalcError("expression is too complex")
    return _eval(tree, text, _new_context())


def format_decimal(value: Decimal) -> str:
    """Plain (non-scientific) string form of a Decimal."""
    if value.is_zero():
        return "0"
    return format(value, "f")


def evaluate(expression: str) -> str:
    """Evaluate ``expression`` and return the result as a plain decimal string."""
    return format_decimal(safe_eval(expression))
