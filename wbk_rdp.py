"""RDP bound from Wang et al.

This uses Google ``dp-accounting`` bound.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Sequence

import numpy as np


# Google's fixed-size event is specified by integer dataset and batch sizes.
# A string or Fraction input is converted to the exact reduced q=b/n supplied by
# the caller; a float is interpreted through its usual decimal string. The main
# experiment passes the original q text, so its sampling rate is not perturbed.
def _sampling_fraction(value: float | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        fraction = value
    elif isinstance(value, str) and "/" in value:
        fraction = Fraction(value)
    else:
        fraction = Fraction(Decimal(str(value)))
    if not 0 < fraction < 1:
        raise ValueError("sampling_probability must lie in (0,1)")
    return fraction


def fswo_replace_one_rdp_profile(
    orders: Sequence[int] | np.ndarray,
    noise_multiplier: float,
    sampling_probability: float | str | Fraction,
) -> np.ndarray:
    """Return one-step Wang et al. RDP at the requested orders."""
    # This function returns one-step RDP values only. The final number of
    # DP-SGD steps and the target privacy delta are applied by rdp_to_dp.py.
    sigma = float(noise_multiplier)
    orders = np.asarray(orders, dtype=float)
    if sigma <= 0.0:
        raise ValueError("noise_multiplier must be positive")
    if np.any(orders <= 1.0):
        raise ValueError("all orders must exceed one")

    try:
        from dp_accounting import dp_event, privacy_accountant
        from dp_accounting.rdp import rdp_privacy_accountant
    except ImportError as exc:
        raise ImportError("Install Google's `dp-accounting` package.") from exc

    # If q=b/n, SampledWithoutReplacementDpEvent models a uniformly chosen
    # fixed batch of size b from n records. Under replace-one adjacency, two
    # clipped sums can differ by 2C / b while the added noise has standard
    # deviation sigma*C / b. Since GaussianDpEvent measures noise relative to
    # unit sensitivity, its effective noise multiplier is therefore sigma/2.
    q = _sampling_fraction(sampling_probability)
    event = dp_event.SampledWithoutReplacementDpEvent(
        source_dataset_size=q.denominator,
        sample_size=q.numerator,
        event=dp_event.GaussianDpEvent(noise_multiplier=sigma / 2.0),
    )
    # Composing the event once asks dp-accounting for the amplified one-step
    # RDP profile. No value is multiplied by the training step count here.
    accountant = rdp_privacy_accountant.RdpAccountant(
        orders=orders,
        neighboring_relation=privacy_accountant.NeighboringRelation.REPLACE_ONE,
    )
    accountant.compose(event)
    # accountant.rdp is aligned entry-by-entry with the supplied orders.
    return np.asarray(accountant.rdp, dtype=float)