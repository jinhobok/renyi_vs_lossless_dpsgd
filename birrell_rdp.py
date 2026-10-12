"""FSRDP implementation, from Birrell et al."""

import numpy as np

from birrell_rdp_bounds import FSwoR_RDP_ro


# This adapter performs no composition. FSwoR_RDP_ro(alpha, sigma, m, q)
# returns Birrell et al.'s one-step replace-one RDP upper bound at order alpha.
# Here q is the fixed batch fraction, sigma is the DP-SGD noise multiplier,
# and m=taylor_order is the first order placed into the bounded Taylor
# remainder (the explicit powers are q^2 through q^(m-1)).
def fswo_replace_one_rdp(
    alpha: float,
    noise_multiplier: float,
    sampling_probability: float,
    *,
    taylor_order: int = 4,
) -> float:
    """Evaluate the upstream one-step bound at one Renyi order."""
    # The returned number is the one-step RDP value rho(alpha).
    return float(
        FSwoR_RDP_ro(
            float(alpha),
            float(noise_multiplier),
            int(taylor_order),
            float(sampling_probability),
        )
    )


def fswo_replace_one_rdp_profile(
    orders: np.ndarray,
    noise_multiplier: float,
    sampling_probability: float,
    *,
    taylor_order: int = 4,
) -> np.ndarray:
    """Evaluate the upstream one-step bound at each requested order."""
    # Evaluate exactly the same one-step formula separately at every requested order.
    return np.asarray(
        [
            FSwoR_RDP_ro(
                float(alpha),
                float(noise_multiplier),
                int(taylor_order),
                float(sampling_probability),
            )
            for alpha in np.asarray(orders, dtype=float)
        ],
        dtype=float,
    )
