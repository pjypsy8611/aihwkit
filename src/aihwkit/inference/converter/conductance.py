# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Conductance converters for the phenomenological noise models for inference."""

from math import log as math_log, sqrt as math_sqrt
from typing import Dict, List, Optional, Tuple

from torch import abs as torch_abs, stack
from torch import Tensor, zeros_like, from_numpy, linspace, allclose, where, randn_like
from torch import ones_like, full_like, bool as torch_bool
from torch import round as torch_round, floor as torch_floor, exp as torch_exp
from torch.autograd import no_grad

from numpy import interp

from aihwkit.inference.converter.base import BaseConductanceConverter

_ZERO_CLIP = 1e-7


@no_grad()
def sample_two_state_conductance(
    mean: float, std: float, distribution: str, like: Tensor
) -> Tensor:
    """Sample programmed conductances of one device state.

    Args:
        mean: mean conductance of the state (linear units)
        std: standard deviation of the programmed conductance (linear units)
        distribution: ``"normal"`` (clamped at zero) or ``"lognormal"``
            (parametrised so that its linear mean and std equal ``mean`` /
            ``std``)
        like: tensor defining shape, dtype and device of the sample

    Returns:
        Tensor of sampled conductances.
    """
    if std <= 0.0:
        return like.new_full(like.shape, mean)
    if distribution == "lognormal":
        sigma2 = math_log(1.0 + (std / mean) ** 2)
        mu_log = math_log(mean) - 0.5 * sigma2
        return torch_exp(mu_log + math_sqrt(sigma2) * randn_like(like))
    return (mean + std * randn_like(like)).clamp(min=0.0)


class SinglePairConductanceConverter(BaseConductanceConverter):
    r"""Single pair of devices.

    Assuming a single pair of devices per cross-point, taking positive
    and negative weights, respectively, where one device is always at
    0.

    Args:
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self, g_max: Optional[float] = None, g_min: Optional[float] = None):
        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min
        self.scale_ratio = None

        if self.g_max < 0.0:
            raise ValueError("g_max should be a positive value")
        if self.g_min < 0.0:
            raise ValueError("g_min should be a positive value")
        if self.g_min >= self.g_max:
            raise ValueError("g_min should be smaller than g_max")

    def __str__(self) -> str:
        return "{}(g_max={:1.2f}, g_min={:1.2f})".format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        abs_max = torch_abs(weights).max()
        scale_ratio = (self.g_max - self.g_min) / abs_max.clamp(min=_ZERO_CLIP)
        scaled_weights = weights * scale_ratio

        conductances = [
            scaled_weights.clamp(min=0.0, max=self.g_max) + self.g_min,
            (-scaled_weights).clamp(min=0.0, max=self.g_max) + self.g_min,
        ]
        params = {"scale_ratio": scale_ratio}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if len(conductances) != 2:
            raise ValueError("conductances must contain exactly two elements")
        if "scale_ratio" not in params:
            raise ValueError("params do not contain scale_ratio")

        weights = ((conductances[0] - self.g_min) - (conductances[1] - self.g_min)) / params[
            "scale_ratio"
        ]

        return weights


class NPairConductanceConverter(BaseConductanceConverter):
    r"""N pairs of conductance devices per unit cell (generalized).

    Assuming a N pairs of devices per cross-point, each having a relative
    weighting (i.e. F factor) as defined by the values in the f_lst parameter.
    For positive and negative weights, one device within in the pair is always
    set to g_min. The higher significant pair will only be used once the
    range of the lower significant pairs is exhausted. In this way, we minimize
    amplification of programming errors and read noise by the scale factors F.
    Note that the scale factors can also be values less than 1.0, however. The
    F factor can be implemented using an amplifying current mirror or by applying
    a longer pulse durations to one pair of conductance pair relative to another,
    such that their is a greater current contribution even though all devices are
    sized equally.

    Args:
        f_lst: In: list of weighting (i.e. scale) factors from lowest
            to hightest, used to determinethe significance of each
            conductance pair.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(
        self, f_lst: List[float], g_max: Optional[float] = None, g_min: Optional[float] = None
    ):

        if not isinstance(f_lst, list):
            raise ValueError("f_lst parameter must be a list of F factors")

        if max(f_lst) < 0.0:
            raise ValueError("f_lst parameter contains negative value")

        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min
        self.f_lst = f_lst
        self.scale_ratio = None

        if self.g_max < 0.0:
            raise ValueError("g_max should be a positive value")
        if self.g_min < 0.0:
            raise ValueError("g_min should be a positive value")
        if self.g_min >= self.g_max:
            raise ValueError("g_min should be smaller than g_max")

    def __str__(self) -> str:
        return "{}(g_max={:1.2f}, g_min={:1.2f})".format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        max_weight_us = sum(f * (self.g_max - self.g_min) for f in self.f_lst)
        max_weight_unitless = torch_abs(weights).max().clamp(min=_ZERO_CLIP)
        scale_ratio = max_weight_us / max_weight_unitless
        weights_us = scale_ratio * weights

        lower_bound_us = 0.0  # lower bound in uS
        conductances = []
        for f_factor in self.f_lst:
            conductances.append(
                ((weights_us.clamp(min=0.0) - lower_bound_us) / f_factor + self.g_min).clamp(
                    min=self.g_min, max=self.g_max
                )
            )  # g_plus
            conductances.append(
                (((-weights_us).clamp(min=0.0) - lower_bound_us) / f_factor + self.g_min).clamp(
                    min=self.g_min, max=self.g_max
                )
            )  # g_minus
            lower_bound_us += f_factor * (self.g_max - self.g_min)

        params = {"scale_ratio": scale_ratio, "f_lst": self.f_lst}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if "f_lst" not in params:
            raise ValueError("params does not contain f_lst")
        if len(conductances) % 2 != 0:
            raise ValueError("unit cell must have an even number of conductances")
        if "scale_ratio" not in params:
            raise ValueError("params does not contain scale_ratio")

        weights = zeros_like(conductances[0])
        for f_factor, (g_plus, g_minus) in zip(
            self.f_lst, zip(conductances[::2], conductances[1::2])
        ):
            weights += f_factor * (g_plus - g_minus)

        return weights / params["scale_ratio"]


class DualPairConductanceConverter(NPairConductanceConverter):
    r"""Two pairs of conductance devices per unit cell (4 devices total).

    Assuming a two pairs of devices per cross-point, each having a relative
    weighting (i.e. F factor) as defined by the values in the f_lst parameter.
    For positive and negative weights, one device within in the pair is always
    set to g_min. The higher significant pair will only be used once the
    range of the lower significant pairs is exhausted. In this way, we minimize
    amplification of programming errors and read noise by the scale factors F.
    Note that the scale factors can also be values less than 1.0, however. The
    F factor can be implemented using an amplifying current mirror or by applying
    a longer pulse durations to one pair of conductance pair relative to another,
    such that their is a greater current contribution even though all devices are
    sized equally.

    Args:
        f_lst: In: list of weighting (i.e. scale) factors from lowest
            to hightest, used to determinethe significance of each
            conductance pair.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(
        self, f_lst: List[float], g_max: Optional[float] = None, g_min: Optional[float] = None
    ):

        if len(f_lst) != 2:
            raise ValueError("f_lst parameter does not contain two values")

        super().__init__(f_lst=f_lst, g_max=g_max, g_min=g_min)


class CustomPairConductanceConverter(BaseConductanceConverter):
    r"""Arbitrary even number of devices.

    Assuming an arbitrary pair of devices per cross-point, each pair having a
    relative weight defined by the values in the f_lst parameter. The parameter
    g_lst is a list of lists that map the unitless weights to a series of
    conductance values. These lists allow us to interpolate and implement a
    function g(w) so that we can map unitless weight to their corresponding
    conductance values. In this way, we can implement very complex conductance
    programming schemes. The various F factors can be implemented using amplifying
    current mirrors or by applying longer pulse durations to the one conductance
    pair relative to others such that their is a greater current contribution even
    though all devices are sized equally.

    Args:
        f_lst: In: list of weighting (i.e. scale) factors used for the
            more significant conductance pairs and the less significant
            conductance pairs.
        g_lst: In: list of lists that map unitless weights to
            conductance values.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(
        self,
        f_lst: List[float],
        g_lst: List[List[float]],
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        invertibility_test: Optional[bool] = True,
    ):
        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min

        if not isinstance(g_lst, list):
            raise ValueError("g_lst parameter must be a list")

        if not all(isinstance(g, list) for g in g_lst):
            raise ValueError("g_lst must be list of lists")

        if len(g_lst) % 2 != 0:
            raise ValueError("g_lst must have and even number of elements")

        if not isinstance(f_lst, list):
            raise ValueError("f_lst parameter must be a list")

        if 2 * len(f_lst) != len(g_lst):
            raise ValueError("must have one value in f_lst for every pair of values in g_lst")

        self.f_lst = f_lst
        self.g_lst = g_lst
        self.scale_ratio = None

        if self.g_max < 0.0:
            raise ValueError("g_max should be a positive value")
        if self.g_min < 0.0:
            raise ValueError("g_min should be a positive value")
        if self.g_min >= self.g_max:
            raise ValueError("g_min should be smaller than g_max")

        if invertibility_test:
            self.invertibility_test()

    def invertibility_test(self) -> None:
        r"""Test to make sure custom conductance converter specification is invertible

        This method tests to make sure the custom converter specification represents
        and invertible function, meaning the g = f(w) <--> w = f^{-1}(x) is true.
        Otherwise, converting unitless weights to conductances and then subsequently
        converting those conductances back to unitless weights will introduce changes
        into the weights, which should not be there. The default is to run this test
        upon instantiation and return an error if it passes. This prevents ill-defined
        custom conductance converter models from corrupting simulation results.
        """
        test_weights = linspace(-1, 1, 5)
        conductances, params = self.convert_to_conductances(test_weights)
        return_weights = self.convert_back_to_weights(conductances, params)
        if not allclose(test_weights, return_weights, atol=0.0001):
            raise ArithmeticError("CustomPairConductanceConverter is not an invertible function")

    def __str__(self) -> str:
        return "{}(g_max={:1.2f}, g_min={:1.2f})".format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:

        weights_us = zeros_like(Tensor(self.g_lst[0])).type_as(weights)
        for f_factor, (gp_lst, gm_lst) in zip(self.f_lst, zip(self.g_lst[::2], self.g_lst[1::2])):
            weights_us += f_factor * (Tensor(gp_lst) - Tensor(gm_lst)).type_as(weights)

        max_weight = torch_abs(weights).max()
        max_weight_us = torch_abs(weights_us).max()
        scale_ratio = max_weight_us / max_weight.clamp(min=_ZERO_CLIP)

        w_lst = (linspace(-max_weight_us, max_weight_us, len(self.g_lst[0])) / scale_ratio).tolist()

        conductances = []
        for f_factor, (gp_lst, gm_lst) in zip(self.f_lst, zip(self.g_lst[::2], self.g_lst[1::2])):
            conductances.append(
                from_numpy(interp(weights.cpu().numpy(), w_lst, gp_lst)).type_as(weights)
            )
            conductances.append(
                from_numpy(interp(weights.cpu().numpy(), w_lst, gm_lst)).type_as(weights)
            )

        params = {"scale_ratio": scale_ratio, "f_lst": self.f_lst}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:

        if "f_lst" not in params:
            raise ValueError("params does not contain f_lst")

        if not isinstance(params["f_lst"], list):
            raise TypeError("f_lst parameter must be a list of f factors")

        if 2 * len(params["f_lst"]) != len(conductances):
            raise ValueError("must have one value in f_lst for every pair of conductances")

        weights_us = zeros_like(conductances[0])
        for f_factor, (g_plus, g_minus) in zip(
            params["f_lst"], zip(conductances[::2], conductances[1::2])
        ):
            weights_us += f_factor * (g_plus - g_minus)

        return weights_us / params["scale_ratio"]  # back to unitless


class SingleDeviceConductanceConverter(BaseConductanceConverter):
    r"""Single devices to represent weights

    Assuming a single bidirectional device per cross-point
    Args:
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self, g_max: Optional[float] = None, g_min: Optional[float] = None):
        self.g_max = 88.199997 if g_max is None else g_max
        self.g_min = 9.0 if g_min is None else g_min
        self.scale_ratio = None

        if self.g_max < 0.0:
            raise ValueError("g_max should be a positive value")
        if self.g_min < 0.0:
            raise ValueError("g_min should be a positive value")
        if self.g_min >= self.g_max:
            raise ValueError("g_min should be smaller than g_max")

    def __str__(self) -> str:
        return "{}(g_max={:1.2f}, g_min={:1.2f})".format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        w_min = weights.min()
        w_max = weights.max()
        scale_ratio = (self.g_max - self.g_min) / (w_max - w_min)
        scaled_weights = (weights - w_min) * scale_ratio
        conductance = scaled_weights + self.g_min
        params = {"scale_ratio": scale_ratio, "min": w_min}
        return conductance, params

    @no_grad()
    def convert_back_to_weights(self, conductances: Tensor, params: Dict) -> Tensor:
        if "scale_ratio" not in params:
            raise ValueError("params do not contain scale_ratio")
        if "min" not in params:
            raise ValueError("params do not contain min")

        if isinstance(conductances, list):
            conductances = stack(conductances)

        weights = params["min"] + ((conductances - self.g_min) / params["scale_ratio"])

        return weights


class BinaryDeviceConductanceConverter(BaseConductanceConverter):
    r"""Two-state (HRS/LRS) devices per weight: signed-magnitude INT weights, one bit per cell.

    **Problem.** The other converters in this module produce a *continuous*
    target conductance ``g = g_min + |w| (g_max - g_min) / max|W|``. A
    device that only has a high-resistance state (HRS) and a
    low-resistance state (LRS), such as a filamentary Ag/MoS2/Au ReRAM
    cell, cannot be programmed to such a value.

    **Solution.** This converter quantises every weight to a
    signed-magnitude integer, slices the magnitude one bit per cell and
    assigns each cell a position weight ``f_k`` (``f_lst``, realised by
    the periphery through shift-and-add or scaled summing). There is no
    sign bit anywhere: the crossbar consists of two physically separate
    arrays of identical shape, a positive and a negative one, each with
    ``n_bits`` cells per weight. The magnitude bits of a positive weight
    are programmed into the positive array (the cells of that weight in
    the negative array stay HRS) and vice versa, and the periphery
    subtracts the two array outputs. This avoids a redundant sign cell
    and lets the mean HRS conductance cancel in the subtraction.

    For a weight ``w`` of a tile with maximum ``max|W|``:

    1. ``q = round(w * L / max|W|)`` with ``L = sum(f_lst)``, an integer
       in ``[-L, L]`` (``2L + 1`` levels);
    2. ``|q| = sum_k f_k b_k`` with bits ``b_k in {0, 1}``;
    3. ``q > 0``: cell ``k`` of the positive array is LRS if ``b_k = 1``;
       ``q < 0``: cell ``k`` of the negative array is LRS if ``b_k = 1``;
       every other cell is HRS.

    Example (``n_bits=4``: 4 magnitude cells per array, ``f = [1, 2, 4,
    8]``, levels ``-15 .. 15``): ``q = 6`` programs the positive array
    cells to ``[0, 1, 1, 0]`` (LSB first) and the negative array cells to
    ``[0, 0, 0, 0]``; ``q = -6`` programs ``[0, 0, 0, 0]`` and
    ``[0, 1, 1, 0]``.

    **Slice layout.** :meth:`convert_to_conductances` returns
    ``[g^+_0, g^-_0, g^+_1, g^-_1, ...]`` (least significant bit first),
    i.e. the positive-array and negative-array cell of every bit
    position. Reading back, ``w = sum_k f_k (g^+_k - g^-_k) /
    scale_ratio`` with ``scale_ratio = (g_lrs - g_hrs) * L / max|W|``, the
    conductance (in :math:`\mu S`) of a unit weight

    **Significance.** ``weighting="binary"`` (default) uses ``f = [1, 2,
    4, ...]`` (bit slicing of an INT magnitude). ``weighting="unary"``
    uses ``f = [1, 1, ...]`` (thermometer code: the level is the number
    of LRS cells).

    **Two's complement (optional).** ``signing="twos_complement"`` codes
    the weight as a two's complement integer in ``n_bits`` single cells
    of one array (the MSB weighs ``-2**(n_bits - 1)``, so here the sign
    is part of the ``n_bits``). For each bit the pair form
    then carries the physical cell on one side and the reference
    conductance ``g_hrs`` on the other (``params["cell_mask"]`` marks the
    physical cells). This coding is included for comparison; it is a poor
    match for analog cells because the MSB error is amplified.

    **Programming noise inside the converter.** With ``g_lrs_std`` or
    ``g_hrs_std`` larger than zero, the conductances returned by
    :meth:`convert_to_conductances` are *programmed* values: a cell
    targeted at LRS is drawn from the LRS distribution (e.g. ``1 ->
    0.98`` in units of ``g_lrs``), a cell targeted at HRS from the HRS
    distribution. The draw is made once per distinct weight matrix and
    cached, so repeated calls with the same weights (e.g. every forward
    pass of a tile) see the same programmed chip; :meth:`reprogram`
    discards the cache. This is what makes the per-cell IR-drop
    simulation of
    :class:`~aihwkit.simulator.configs.configs.TorchInferenceRPUConfigIRDropT`
    work on noisy binary cells: that tile calls
    :meth:`convert_to_conductances` on its weights at every forward pass
    and computes the Thevenin-equivalent IR drop for each pair of
    positive/negative array columns (with significance ``f_lst``)
    separately. Use either this mechanism or the programming noise of
    :class:`~aihwkit.inference.noise.reram.TwoStateReRamNoiseModel`, not
    both.

    Args:
        n_bits: number of magnitude cells per weight in each of the two
            arrays (``2 * n_bits`` cells per weight in total). Default 4,
            i.e. ``f = [1, 2, 4, 8]`` and 31 levels.
        weighting: ``"binary"`` (default) or ``"unary"`` significance of
            the magnitude cells. Ignored for ``"twos_complement"``.
        signing: ``"sign_magnitude"`` (default; ``"differential"`` is
            accepted as an alias) or ``"twos_complement"``.
        n_pairs: alias of ``n_bits`` (AIHWKIT pair terminology: one cell
            in the positive and one in the negative array per bit).
        g_lrs: mean LRS ("on") conductance in :math:`\mu S`.
        g_hrs: mean HRS ("off") conductance in :math:`\mu S`.
        f_lst: explicit list of positive integer significance factors of
            the magnitude cells, least significant first (e.g.
            ``[1, 2, 4]``). Must allow a greedy decomposition of every
            integer up to ``sum(f_lst)``. Not allowed for
            ``"twos_complement"``.
        g_lrs_std: standard deviation of the programmed LRS conductance
            in :math:`\mu S` (0 disables programming noise).
        g_hrs_std: standard deviation of the programmed HRS conductance.
        distribution: ``"lognormal"`` or ``"normal"`` programming
            distribution.

    Attributes:
        f_lst: significance factors of the magnitude cells.
        pair_f_lst: significance factors of the returned conductance
            pairs (``f_lst`` plus the MSB weight for two's complement).
        g_max: alias of ``g_lrs`` (for code that expects ``g_max``).
        g_min: alias of ``g_hrs``.
    """

    # pylint: disable=too-many-instance-attributes

    SIGNINGS = ("sign_magnitude", "twos_complement")

    def __init__(  # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
        self,
        n_bits: Optional[int] = None,
        weighting: str = "binary",
        *,
        signing: Optional[str] = None,
        n_pairs: Optional[int] = None,
        g_lrs: float = 100.0,
        g_hrs: float = 1.0,
        f_lst: Optional[List[float]] = None,
        g_lrs_std: float = 0.0,
        g_hrs_std: float = 0.0,
        distribution: str = "lognormal",
    ):
        if signing is None or signing == "differential":
            signing = "sign_magnitude"
        if signing not in self.SIGNINGS:
            raise ValueError("signing must be one of {}".format(self.SIGNINGS))
        if weighting not in ("unary", "binary"):
            raise ValueError("weighting must be 'unary' or 'binary'")

        def significance(n_cells: int) -> List[float]:
            if weighting == "unary":
                return [1.0] * n_cells
            return [float(2**k) for k in range(n_cells)]

        msb_weight = 0.0
        if signing == "sign_magnitude":
            if f_lst is None:
                if n_pairs is not None and n_bits is not None and n_pairs != n_bits:
                    raise ValueError("n_pairs is an alias of n_bits; give only one of them")
                n_bits = n_pairs if n_bits is None else n_bits
                n_bits = 4 if n_bits is None else n_bits
                if n_bits < 1:
                    raise ValueError("need at least one magnitude cell per array")
                f_lst = significance(n_bits)
            else:
                weighting = "custom"
        else:  # twos_complement
            if f_lst is not None:
                raise ValueError("f_lst cannot be given for twos_complement")
            n_bits = 4 if n_bits is None else n_bits
            if n_bits < 2:
                raise ValueError("twos_complement needs n_bits >= 2")
            weighting = "binary"
            f_lst = [float(2**k) for k in range(n_bits - 1)]
            msb_weight = float(2 ** (n_bits - 1))

        if any(f <= 0 or float(f) != int(f) for f in f_lst):
            raise ValueError("f_lst must contain positive integers")
        if g_hrs < 0.0:
            raise ValueError("g_hrs should be a non-negative value")
        if g_lrs <= g_hrs:
            raise ValueError("g_lrs must be larger than g_hrs")
        if distribution not in ("normal", "lognormal"):
            raise ValueError("distribution must be 'normal' or 'lognormal'")

        self.f_lst = [float(f) for f in f_lst]
        self.pair_f_lst = self.f_lst + ([msb_weight] if msb_weight else [])
        self.signing = signing
        self.weighting = weighting
        self.n_pairs = len(self.f_lst)  # magnitude cells per array
        self.n_bits = len(self.f_lst) + (1 if signing == "twos_complement" else 0)
        self.g_lrs = float(g_lrs)
        self.g_hrs = float(g_hrs)
        self.g_max = self.g_lrs
        self.g_min = self.g_hrs
        # threshold separating the two states
        self.g_threshold = (self.g_lrs * self.g_hrs) ** 0.5 if g_hrs > 0 else 0.5 * self.g_lrs
        self.g_lrs_std = float(g_lrs_std)
        self.g_hrs_std = float(g_hrs_std)
        self.distribution = distribution
        self._programmed: Dict[Tuple, List[Tensor]] = {}

    def __eq__(self, other: object) -> bool:
        if self.__class__ != other.__class__:
            return False
        mine = {k: v for k, v in self.__dict__.items() if k != "_programmed"}
        theirs = {k: v for k, v in other.__dict__.items() if k != "_programmed"}
        return mine == theirs

    def __str__(self) -> str:
        text = "{}(signing={}, n_bits={}, weighting={}, f_lst={}, g_lrs={:1.3g}, g_hrs={:1.3g})"
        return text.format(
            self.__class__.__name__,
            self.signing,
            self.n_bits,
            self.weighting,
            self.f_lst,
            self.g_lrs,
            self.g_hrs,
        )

    @property
    def max_level(self) -> int:
        """Largest representable integer level ``L = sum(f_lst)``."""
        return int(round(sum(self.f_lst)))

    @property
    def n_levels(self) -> int:
        """Number of distinct weight levels, ``2 L + 1`` (zero included)."""
        return 2 * self.max_level + 1

    @property
    def n_devices(self) -> int:
        """Number of physical two-state devices per weight.

        Sign-magnitude: ``n_bits`` cells in the positive array plus the
        same number in the negative array. Two's complement: ``n_bits``
        cells.
        """
        if self.signing == "sign_magnitude":
            return 2 * self.n_pairs
        return self.n_bits

    @property
    def n_slices(self) -> int:
        """Number of conductance tensors returned by :meth:`convert_to_conductances`."""
        return 2 * len(self.pair_f_lst)

    @property
    def hwa_res(self) -> float:
        """Resolution ``1 / L`` for ``WeightModifierType.DISCRETIZE``.

        With ``rel_to_actual_wmax=True`` the (RPUCuda) inference tile
        rounds ``w / max|W|`` to integer multiples of ``res``, so this
        value makes the training grid coincide with the ``2 L + 1``
        levels of the unit cell.

        Note:
            The pure-torch tile uses the convention ``res * 2 * max|W|``;
            use ``1 / (2 L)`` there.
        """
        return 1.0 / self.max_level

    @property
    def has_programming_noise(self) -> bool:
        """Whether programmed (noisy) conductances are returned."""
        return self.g_lrs_std > 0.0 or self.g_hrs_std > 0.0

    def reprogram(self) -> None:
        """Forget all programmed conductances (the next conversion re-draws them)."""
        self._programmed = {}

    @no_grad()
    def quantize(self, weights: Tensor) -> Tuple[Tensor, Tensor]:
        """Weights -> integer levels ``q`` in ``[-L, L]`` and ``level_scale = L / max|W|``."""
        abs_max = weights.abs().max().clamp(min=_ZERO_CLIP)
        level_scale = self.max_level / abs_max
        q_levels = torch_round(weights * level_scale).clamp(-self.max_level, self.max_level)
        return q_levels, level_scale

    @no_grad()
    def _decompose(self, magnitude: Tensor) -> List[Tensor]:
        """Non-negative integer levels -> one ``{0, 1}`` tensor per magnitude device (LSB first)."""
        if self.weighting == "unary":
            return [(magnitude > k).to(magnitude.dtype) for k in range(len(self.f_lst))]

        remaining = magnitude.clone()
        bits = [zeros_like(magnitude) for _ in self.f_lst]
        for idx in reversed(range(len(self.f_lst))):
            bit = (remaining >= self.f_lst[idx]).to(magnitude.dtype)
            remaining -= bit * self.f_lst[idx]
            bits[idx] = bit
        if remaining.abs().max() > 0:
            raise ArithmeticError(
                "f_lst {} cannot represent all levels up to {}".format(self.f_lst, self.max_level)
            )
        return bits

    @no_grad()
    def quantized_weights(self, weights: Tensor) -> Tensor:
        """Ideal (noise-free) weights after the two-state mapping."""
        conductances, params = self._ideal_conductances(weights)
        return self.convert_back_to_weights(conductances, params)

    def _bits_to_conductance(self, bits: Tensor) -> Tensor:
        return self.g_hrs + bits * (self.g_lrs - self.g_hrs)

    @no_grad()
    def _ideal_conductances(  # pylint: disable=too-many-locals
        self, weights: Tensor
    ) -> Tuple[List[Tensor], Dict]:
        """Pair-form target conductances (no programming noise) and params."""
        q_levels, level_scale = self.quantize(weights)
        conductances: List[Tensor] = []
        cell_mask: List[Tensor] = []

        if self.signing == "sign_magnitude":
            # magnitude bits go to the positive array for q > 0 and to the
            # negative array for q < 0; the other array holds HRS cells
            bits_plus = self._decompose(q_levels.clamp(min=0))
            bits_minus = self._decompose((-q_levels).clamp(min=0))
            for b_plus, b_minus in zip(bits_plus, bits_minus):
                conductances += [
                    self._bits_to_conductance(b_plus),
                    self._bits_to_conductance(b_minus),
                ]
                cell_mask += [ones_like(b_plus, dtype=torch_bool)] * 2
        else:  # twos_complement
            modulus = float(2**self.n_bits)
            code = where(q_levels < 0, q_levels + modulus, q_levels)
            reference = full_like(q_levels, self.g_hrs)
            true_mask = ones_like(q_levels, dtype=torch_bool)
            for k in range(self.n_bits):
                bit = torch_floor(code / float(2**k)).remainder(2.0)
                g_cell = self._bits_to_conductance(bit)
                if k < self.n_bits - 1:
                    conductances += [g_cell, reference]
                    cell_mask += [true_mask, ~true_mask]
                else:  # MSB has negative weight: it sits on the minus side
                    conductances += [reference, g_cell]
                    cell_mask += [~true_mask, true_mask]

        params = {
            "scale_ratio": level_scale * (self.g_lrs - self.g_hrs),
            "level_scale": level_scale,
            "f_lst": self.pair_f_lst,
            "cell_mask": cell_mask,
            "g_lrs": self.g_lrs,
            "g_hrs": self.g_hrs,
            "signing": self.signing,
        }
        return conductances, params

    @no_grad()
    def _program(self, key: Tuple, ideal: List[Tensor], cell_mask: List[Tensor]) -> List[Tensor]:
        """Return the programmed (noisy) conductances for the ideal HRS/LRS targets."""
        if key in self._programmed:
            return self._programmed[key]
        programmed = []
        for g_target, mask in zip(ideal, cell_mask):
            is_lrs = g_target > self.g_threshold
            g_lrs = sample_two_state_conductance(
                self.g_lrs, self.g_lrs_std, self.distribution, g_target
            )
            g_hrs = sample_two_state_conductance(
                self.g_hrs, self.g_hrs_std, self.distribution, g_target
            )
            programmed.append(where(mask, where(is_lrs, g_lrs, g_hrs), g_target))
        self._programmed[key] = programmed
        return programmed

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        """Weights -> ``[g^+_0, g^-_0, g^+_1, g^-_1, ...]`` (least significant pair first).

        Every entry is exactly ``g_hrs`` or ``g_lrs`` unless programming
        noise is enabled, in which case the physical devices (see
        ``params["cell_mask"]``) carry their programmed conductances.
        """
        conductances, params = self._ideal_conductances(weights)
        if self.has_programming_noise:
            key = (
                tuple(weights.shape),
                str(weights.device),
                round(float(weights.sum()), 4),
                round(float(weights.square().sum()), 4),
            )
            conductances = self._program(key, conductances, params["cell_mask"])
        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if "scale_ratio" not in params:
            raise ValueError("params do not contain scale_ratio")
        if len(conductances) != self.n_slices:
            raise ValueError(
                "expected {} conductance slices, got {}".format(self.n_slices, len(conductances))
            )
        weights = zeros_like(conductances[0])
        for f_factor, (g_plus, g_minus) in zip(
            self.pair_f_lst, zip(conductances[::2], conductances[1::2])
        ):
            weights += f_factor * (g_plus - g_minus)
        return weights / params["scale_ratio"]
