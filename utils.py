"""Shared exact-input, precision, and display helpers for Arb calculations."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, localcontext
from fractions import Fraction
from operator import index
from typing import Any, Iterator, TypeAlias

from flint import arb, ctx

import math

Scalar: TypeAlias = int | float | str | arb


# Exact input conversion matters because all later Arb endpoints are only as
# meaningful as the supplied parameters. A string such as "1/100" remains
# exactly rational; a Python float is converted through its exact binary ratio
# rather than through a shortened decimal rendering.
def arb_input(value: Scalar) -> arb:
    """Create an Arb value without silently shortening a Python float."""
    if isinstance(value, arb):
        return value
    if isinstance(value, float):
        numerator, denominator = value.as_integer_ratio()
        return arb(f"{numerator}/{denominator}")
    return arb(str(value))


# This parser accepts the common exact representations while
# rejecting an interval or rational value that merely lies near an integer.
def integer_order(value: Any, minimum: int = 2) -> int:
    """Parse an exact integer and enforce a minimum order."""
    try:
        order = index(value)
    except TypeError:
        if isinstance(value, float) and value.is_integer():
            order = int(value)
        elif isinstance(value, arb) and value.is_exact():
            exact_integer = value.unique_fmpz()
            if exact_integer is None:
                raise ValueError("alpha must be an integer")
            order = int(exact_integer)
        elif isinstance(value, str):
            fraction = Fraction(value.strip())
            if fraction.denominator != 1:
                raise ValueError("alpha must be an integer")
            order = fraction.numerator
        else:
            raise ValueError("alpha must be an integer")
    if order < minimum:
        raise ValueError(f"alpha must be an integer at least {minimum}")
    return int(order)


# FLINT accepts numerator/denominator text, so this is the lossless bridge from
# Python's Fraction parameters to Arb arithmetic.
def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# Exact Arb endpoints are dyadic rationals. Recovering that rational exactly
# is needed when heap priorities or tolerance allocations are represented by
# Python Fraction objects outside the Arb context.
def exact_arb_fraction(value: arb) -> Fraction:
    """Convert an exact finite Arb endpoint to a rational number."""
    if not value.is_exact() or not value.is_finite():
        raise ValueError("expected an exact finite Arb endpoint")
    mantissa, exponent = value.man_exp()
    mantissa = int(mantissa)
    exponent = int(exponent)
    if exponent >= 0:
        return Fraction(mantissa * (1 << exponent), 1)
    return Fraction(mantissa, 1 << -exponent)


# mid_rad_10exp returns a decimal enclosure. Adding the radius for an upper
# endpoint and subtracting it for a lower endpoint preserves outward rounding
# when the result is written as ordinary fixed-point text.
def upper_text(value: arb, digits: int = 30) -> str:
    midpoint, radius, exponent = value.upper().mid_rad_10exp(max(1, int(digits)))
    return fixed_decimal_text(int(midpoint + radius), int(exponent))


def lower_text(value: arb, digits: int = 30) -> str:
    midpoint, radius, exponent = value.lower().mid_rad_10exp(max(1, int(digits)))
    return fixed_decimal_text(int(midpoint - radius), int(exponent))


def upper_text_with_tolerance(
    upper: arb,
    lower: arb,
    tolerance: arb,
    digits: int = 30,
) -> str:
    # A displayed upper bound is useful only if decimal rounding does not make
    # the displayed gap exceed the requested certificate tolerance. Increase
    # the number of decimal digits until
    #
    #   displayed upper - certified lower <= tolerance.
    """Format an upper endpoint finely enough to retain the certified gap."""
    lower_fraction = exact_arb_fraction(lower)
    tolerance_fraction = exact_arb_fraction(tolerance)
    digits = max(1, int(digits))
    while True:
        midpoint, radius, exponent = upper.mid_rad_10exp(digits)
        mantissa = int(midpoint + radius)
        exponent = int(exponent)
        decimal_upper = Fraction(mantissa) * Fraction(10) ** exponent
        if decimal_upper - lower_fraction <= tolerance_fraction:
            return fixed_decimal_text(mantissa, exponent)
        digits *= 2


# This helper is for readable diagnostics such as the best dyadic u. The
# enlarged Decimal precision prevents the numerator or denominator size from
# silently limiting the requested number of digits.
def fraction_decimal_text(value: Fraction, digits: int = 30) -> str:
    if value == 0:
        return "0"
    with localcontext() as context:
        context.prec = max(
            int(digits),
            len(str(abs(value.numerator))) + len(str(value.denominator)) + 10,
        )
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


# FLINT uses one precision setting for the whole Python process.
# A certified calculation may temporarily increase that precision.
# This context manager saves the previous value and restores it afterward,
# so calling one certified routine does not change the precision used by
# subsequent routines.
@contextmanager 
def restore_precision() -> Iterator[None]:
    """Restore FLINT's process-global decimal precision after a calculation."""
    previous = ctx.dps
    try:
        yield
    finally:
        ctx.dps = previous

"""Small formatting helpers shared by numerical modules."""




# Place the decimal point directly in the integer mantissa. Avoiding scientific
# notation makes certified upper endpoints easy to pass back into exact parsers.
def fixed_decimal_text(mantissa: int, exponent: int) -> str:
    """Format ``mantissa * 10**exponent`` without scientific notation."""
    if mantissa == 0:
        return "0"
    sign = "-" if mantissa < 0 else ""
    digits = str(abs(int(mantissa)))
    point = len(digits) + int(exponent)
    if point <= 0:
        return f"{sign}0.{('0' * -point)}{digits}"
    if point >= len(digits):
        return f"{sign}{digits}{'0' * (point - len(digits))}"
    return f"{sign}{digits[:point]}.{digits[point:]}"


# A nearest binary-float conversion can lie just below an exact decimal upper
# bound. Move by one representable float precisely in that case.
def upward_float(decimal_text: str) -> float:
    """Convert an exact decimal upper bound to a binary float without rounding down."""
    target = Decimal(decimal_text)
    value = float(target)
    return math.nextafter(value, math.inf) if Decimal.from_float(value) < target else value
