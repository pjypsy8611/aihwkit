# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Phenomenological noise models for ReRAM devices for inference."""

from copy import deepcopy
from typing import Any, List, Optional, Dict, Sequence, Tuple, cast

import numpy as np
from torch import randn_like, rand_like, stack, where, Tensor
from torch.autograd import no_grad
from numpy import log, log10, sqrt
from aihwkit.exceptions import ArgumentError
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
    SingleDeviceConductanceConverter,
    BinaryDeviceConductanceConverter,
    sample_two_state_conductance,
)


class ReRamWan2022NoiseModel(BaseNoiseModel):
    r"""Noise model that was inferred from ReRam publication data.

    This ReRam model is and approximation to the data published by
    `Wan et al. Nature (2022)`_.

    Conductance dependence of the deviations from the target
    conductance was estimated from the published figures and fitted
    with a 4-th order polynomial (only 1 sec, 1 day, 2 day).

    No separate data is available for read noise (1/f).

    Note:

        To account for short-term read noise (about 1\%) one should
        additional set the ``forward.w_noise`` parameter to about 0.01
        (with w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    Args:

        coeff_dic: polynomial coefficients in :math:`\mu S`,
            :math:`\sum_i c_i \left(\frac{g_t}{g_\max}\right)^i` for
            each time. If not given, the fitted measurement is taken
            at selected time points only

        g_converter: Instantiated class of the conductance converter
            (defaults to single pair).
        g_max: In :math:`\mu S`, the maximal conductance, i.e. the value
            the absolute max of the weights will be mapped to.
        noise_scale: Additional scale for the noise.
        coeff_g_max_reference: reference :math:`g_\max` value
            when fitting the coefficients, since the result of the
            polynomial fit is given in uS. If
            ``coeff_g_max_reference`` is not given and
            `coeffs` are given explicitely, it will be set to
            ``g_max`` of the conductance converter.

    .. _`Wan et al. Nature (2022)`: https://www.nature.com/articles/s41586-022-04992-8

    """

    def __init__(
        self,
        coeff_dic: Optional[Dict[float, List]] = None,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = None,
        noise_scale: float = 1.0,
        coeff_g_max_reference: Optional[float] = None,
    ):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)

        self.g_max = getattr(self.g_converter, "g_max", g_max)
        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")

        if coeff_g_max_reference is None:
            self.coeff_g_max_reference = self.g_max

        if coeff_dic is None:
            # standard g_max are defined in respect to 40.0 uS. Need to
            # adjust for that in case g_max is not equal to 40.0 uS

            coeff_dic = {
                1.0: [-16.815, 45.393, -43.853, 16.030, 0.348][::-1],
                3600 * 24.0: [-16.458, 47.095, -50.773, 22.086, 0.701][::-1],
                3600 * 24.0 * 2: [-11.934, 37.062, -43.507, 20.274, 0.782][::-1],
            }
            self.prog_coeff_g_max_reference = 40.0
        self.coeff_dic = coeff_dic
        self.noise_scale = noise_scale

    def _apply_poly(self, g_target: Tensor, coeff: List, scale: float = 1.0) -> Tensor:
        """Applied polynomial noise"""

        mat = 1
        sig_prog = coeff[0]
        for value in coeff[1:]:
            mat *= g_target / self.g_max
            sig_prog += mat * value

        sig_prog *= self.g_max / self.coeff_g_max_reference  # type: ignore
        g_prog = g_target + scale * sig_prog * randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 2-degree
        polynomial.
        """

        min_key = min(list(self.coeff_dic.keys()))
        return self._apply_poly(g_target, self.coeff_dic[min_key], self.noise_scale)

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return target values as coefficients.

        Since ReRAM does not show drift in the usual sense, here
        simply the target values will given as coefficients to compute
        the long-term variations on-the-fly

        """
        return g_target

    @no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: Tensor, g_target: Tensor, t_inference: float
    ) -> Tensor:
        """Apply the accumulated noise according to the time of inference.

        Will use unique 4th-order polynomial fit to the ReRAM
        measurements to the target values.

        Args:
            g_prog: will be ignored
            g_target: target conductance values that will be used to add noise
            t_inference: time of inference.

        Returns:
            conductances with noise applied

        Raises:
            ArgumentError: if `t_inference` is not one of
                ``(1, 24*3600, 2*24*3600)`` seconds (or any user-defined
                key in ``coeff_dic``), the error will be raised.
        """
        # pylint: disable=arguments-renamed

        if t_inference not in self.coeff_dic:
            raise ArgumentError(f"t_inference should be one of `{list(self.coeff_dic.keys())}`")

        g_final = self._apply_poly(g_target, self.coeff_dic[t_inference], self.noise_scale)

        return g_final.clamp(min=0.0)


class ReRamCMONoiseModel(BaseNoiseModel):
    r"""Noise model inferred from Analog Filamentary Conductive-Metal-Oxide
    (CMO)/HfOx ReRAM devices from IBM Research Europe - Zurich.

    This noise model is estimated from statistical characterization of CMO/HfOx devices from
    Falcone et al. (In Review)

    Programming noise:
        Described by a linear function with respect to the G target.
        Coefficients are considered for two acceptance ranges, 0.2% and 2% of target conductance

    Conductance Decay:
        Drift in CMO/HfOx devices showed independence of the target conductance value.
        Mean and STD of the conductance distribution were fitted with 1st-order polynomial
        as a function of the log(t) where t is the time of inference

    TODO:
    Read noise (1/f) characterization of CMO/HfO<sub>x</sub> available at Lombardo et al. DRC
    (2024) but not implemented.

    Note:

        To account for short-term read noise (about 1\%) one should
        additional set the ``forward.w_noise`` parameter to about 0.01
        (with w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    Args:
        coeff_dict:  acceptance range with coefficients for the programming noise in :math:`\mu S`,
        g_converter: Instantiated class of the conductance converter for a single device per
        cross-point.
        g_max: In :math:`\mu S`, the maximal conductance, i.e. the value the absolute max of
        the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, i.e. the value the absolute min of
        the weights will be mapped to.
        prog_noise_scale: Scale for the programming noise.
        read_noise_scale: Scale for the read and accumulated noise.
        drift_scale: Scale for the  drift coefficient.
        since the result of the polynomial fit is given in uS.
        decay_dict: mean and std coefficients for the drift noise in :math:`\mu S`,

    """

    def __init__(
        self,
        coeff_dict: Optional[Dict[float, List]] = None,
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        prog_noise_scale: float = 1.0,
        read_noise_scale: float = 1.0,
        drift_scale: float = 1.0,
        decay_dict: Optional[Dict[str, List]] = None,
        read_dict: Optional[Dict[str, float]] = None,
        acceptance_range: float = 2e-2,
    ):
        g_converter = SingleDeviceConductanceConverter(g_max=g_max, g_min=g_min)
        super().__init__(g_converter)
        g_max = getattr(self.g_converter, "g_max", g_max)
        g_min = getattr(self.g_converter, "g_min", g_min)
        if g_max is None:
            raise ValueError("g_max cannot be established from g_converter")
        if g_min is None:
            raise ValueError("g_min cannot be established from g_converter")
        self.g_max = g_max
        self.g_min = g_min
        self.coeff_g_max_reference = self.g_max
        if coeff_dict is None:
            coeff_dict = {
                0.2: [0.00106879, 0.00081107][::-1],
                2: [0.01129027418, 0.0112185391][::-1],
            }
        if read_dict is None:
            read_dict = {"K": 0.0277316483, "t_read": 1e-6}
        if decay_dict is None:
            decay_dict = {"mean": [-0.08900206], "std": [0.04201137, 0.41183342]}
        if acceptance_range not in coeff_dict.keys():
            acceptance_range = min(coeff_dict.keys())
        self.coeff_dict = coeff_dict
        self.prog_noise_scale = prog_noise_scale
        self.read_noise_scale = read_noise_scale
        self.drift_scale = drift_scale
        self.decay_dict = decay_dict
        self.read_dict = read_dict
        self.acceptance_range = acceptance_range

    def _apply_poly(self, g_target: Tensor, coeff: List, scale: float = 1.0) -> Tensor:
        """Applied polynomial noise"""
        mat = 1
        sig_prog = coeff[0]
        for value in coeff[1:]:
            mat *= g_target  # / self.g_max
            sig_prog += mat * value
        sig_prog *= self.g_max / self.coeff_g_max_reference
        g_prog = g_target + sig_prog * randn_like(g_target) * scale
        return g_prog

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 1st-degree
        polynomial.
        Depends of the acceptance range of the program-and-verify loop
        """

        min_key = (
            self.acceptance_range
            if self.acceptance_range in self.coeff_dict.keys()
            else min(list(self.coeff_dict.keys()))
        )
        return self._apply_poly(g_target, self.coeff_dict[min_key], self.prog_noise_scale)

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Conductance relaxation is independent of the conductance level"""
        return g_target

    @no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: Tensor, drift_noise_param: Tensor, t_inference: float
    ) -> Tensor:
        """Apply the accumulated noise according to the time of inference.

        Will use unique 1st-order polynomial fits the conductance mean shift
        and standard deviation shift from the ReRAM

        Args:
            g_prog: target conductance values that will be used to add noise
            drift_noise_param: ccoefficients of the mean and std drift
            t_inference: time of inference. Times in seconds

        Returns:
            conductances with noise applied

        """
        if t_inference == 0:
            return g_prog

        g_mean = g_prog + (self.decay_dict["mean"][0] * log(t_inference) * self.drift_scale)

        sigma_relaxation = self.decay_dict["std"][0] * log(t_inference) + self.decay_dict["std"][1]
        g_drift = g_mean + (randn_like(g_prog) * sigma_relaxation * self.drift_scale)
        sigma_read = (
            self.read_dict["K"]
            * log10(g_drift)
            * sqrt(log((t_inference + self.read_dict["t_read"]) / (2 * self.read_dict["t_read"])))
        )
        g_final = g_drift + sigma_read * randn_like(g_prog) * self.read_noise_scale

        return g_final.clamp(min=self.g_min, max=self.g_max)


class TwoStateReRamNoiseModel(BaseNoiseModel):
    r"""Noise model for two-state (HRS/LRS) ReRAM devices.

    Filamentary devices such as Ag/MoS2/Au switch between a
    high-resistance state (HRS) and a low-resistance state (LRS) and
    cannot be tuned to intermediate conductances. The analog models in
    this module describe the programming error as a polynomial in the
    *continuous* target conductance, which does not apply. This model
    instead describes each device **per state**, with parameters that
    can be read off a measured device population (see
    :meth:`from_measurements`). It is meant to be used together with
    :class:`~aihwkit.inference.converter.conductance.BinaryDeviceConductanceConverter`,
    which only ever produces ``g_hrs`` / ``g_lrs`` targets.

    Notation: conductances in :math:`\mu S`, times in seconds after
    programming, :math:`\xi` a standard normal number per device,
    :math:`U` uniform in :math:`[0, 1)`.

    **Programming**

    (P1) the target state is ``LRS`` if :math:`g_T > \sqrt{g_{hrs} g_{lrs}}`.

    (P2) switching failure: an LRS target stays HRS with probability
    ``p_set_fail``; an HRS target stays LRS with ``p_reset_fail``.

    (P3) the programmed conductance is drawn from the distribution of the
    *realised* state with (mean, std) = (``g_lrs``, ``g_lrs_std``) or
    (``g_hrs``, ``g_hrs_std``), either normal (clamped at 0) or
    log-normal with matched linear mean and std.

    **Retention and read**

    (R1) power law per device, :math:`g(t) = g_{prog} ((t + t_0)/t_0)^{-\nu}`
    with :math:`\nu = \mathrm{drift\_scale}\,(\nu_{mean} + \nu_{std}\,\xi)`
    of the realised state (``nu_lrs_*`` or ``nu_hrs_*``).

    (R2) spontaneous state loss with probability
    :math:`1 - (1 - p)^{d}` after :math:`d = \log_{10}((t + t_0)/t_0)`
    decades (``p_retention_fail_lrs`` / ``p_retention_fail_hrs``); a lost
    device is redrawn from the other state (P3).

    (R3) multiplicative read noise with relative std ``read_noise_rel_*``
    of the state and the 1/f accumulation factor
    :math:`\sqrt{\ln((t + t_0 + t_{read}) / (2 t_{read}))}`.

    Note:
        :meth:`apply_drift_noise` is overridden. The base implementation
        re-derives the device conductances from the *programmed weights*
        through the converter. With a quantising converter this would
        snap every device back to the ideal HRS/LRS value and erase the
        programming noise. This model therefore stores
        ``[g_prog, nu_lrs, nu_hrs, scale_ratio]`` per device slice in the
        drift parameters returned by :meth:`apply_programming_noise`
        (plus a mask of the physical devices) and drifts those stored
        conductances; reference entries of the converter stay exact.

    Args:
        g_converter: unit cell (defaults to one pair of two-state devices
            per weight).
        g_lrs: mean LRS conductance (defaults to the converter's).
        g_hrs: mean HRS conductance (defaults to the converter's).
        g_lrs_std: std of the programmed LRS conductance (P3).
        g_hrs_std: std of the programmed HRS conductance (P3).
        distribution: ``"lognormal"`` or ``"normal"`` (P3).
        p_set_fail: SET failure probability (P2).
        p_reset_fail: RESET failure probability (P2).
        prog_noise_scale: multiplier on both stds.
        nu_lrs_mean: mean retention exponent of LRS (R1).
        nu_lrs_std: std of the LRS retention exponent (R1).
        nu_hrs_mean: mean retention exponent of HRS (R1).
        nu_hrs_std: std of the HRS retention exponent (R1).
        p_retention_fail_lrs: LRS -> HRS loss probability per decade (R2).
        p_retention_fail_hrs: HRS -> LRS loss probability per decade (R2).
        drift_scale: multiplier on both exponents.
        t_0: time of the first read after programming (R1).
        read_noise_rel_lrs: relative read-noise std in LRS (R3).
        read_noise_rel_hrs: relative read-noise std in HRS (R3).
        read_noise_scale: multiplier on the read noise.
        t_read: read duration for the 1/f accumulation (R3).
    """

    # pylint: disable=too-many-arguments

    def __init__(  # pylint: disable=too-many-locals
        self,
        g_converter: Optional[BinaryDeviceConductanceConverter] = None,
        *,
        g_lrs: Optional[float] = None,
        g_hrs: Optional[float] = None,
        g_lrs_std: float = 0.0,
        g_hrs_std: float = 0.0,
        distribution: str = "lognormal",
        p_set_fail: float = 0.0,
        p_reset_fail: float = 0.0,
        prog_noise_scale: float = 1.0,
        nu_lrs_mean: float = 0.0,
        nu_lrs_std: float = 0.0,
        nu_hrs_mean: float = 0.0,
        nu_hrs_std: float = 0.0,
        p_retention_fail_lrs: float = 0.0,
        p_retention_fail_hrs: float = 0.0,
        drift_scale: float = 1.0,
        t_0: float = 20.0,
        read_noise_rel_lrs: float = 0.0,
        read_noise_rel_hrs: float = 0.0,
        read_noise_scale: float = 1.0,
        t_read: float = 250.0e-9,
    ):
        if g_converter is None:
            g_converter = BinaryDeviceConductanceConverter(
                n_pairs=1,
                g_lrs=100.0 if g_lrs is None else g_lrs,
                g_hrs=1.0 if g_hrs is None else g_hrs,
            )
        else:
            g_converter = deepcopy(g_converter)
        super().__init__(g_converter)

        self.g_lrs = float(g_converter.g_lrs if g_lrs is None else g_lrs)
        self.g_hrs = float(g_converter.g_hrs if g_hrs is None else g_hrs)
        if self.g_lrs <= self.g_hrs:
            raise ValueError("g_lrs must be larger than g_hrs")
        if distribution not in ("normal", "lognormal"):
            raise ValueError("distribution must be 'normal' or 'lognormal'")
        for name, prob in (
            ("p_set_fail", p_set_fail),
            ("p_reset_fail", p_reset_fail),
            ("p_retention_fail_lrs", p_retention_fail_lrs),
            ("p_retention_fail_hrs", p_retention_fail_hrs),
        ):
            if not 0.0 <= prob <= 1.0:
                raise ValueError("{} must be a probability".format(name))

        self.g_lrs_std = float(g_lrs_std)
        self.g_hrs_std = float(g_hrs_std)
        self.distribution = distribution
        self.p_set_fail = float(p_set_fail)
        self.p_reset_fail = float(p_reset_fail)
        self.prog_noise_scale = float(prog_noise_scale)
        self.nu_lrs_mean = float(nu_lrs_mean)
        self.nu_lrs_std = float(nu_lrs_std)
        self.nu_hrs_mean = float(nu_hrs_mean)
        self.nu_hrs_std = float(nu_hrs_std)
        self.p_retention_fail_lrs = float(p_retention_fail_lrs)
        self.p_retention_fail_hrs = float(p_retention_fail_hrs)
        self.drift_scale = float(drift_scale)
        self.t_0 = float(t_0)
        self.read_noise_rel_lrs = float(read_noise_rel_lrs)
        self.read_noise_rel_hrs = float(read_noise_rel_hrs)
        self.read_noise_scale = float(read_noise_scale)
        self.t_read = float(t_read)

        # (P1) state decision threshold: geometric mean of the two states
        self.g_threshold = sqrt(self.g_lrs * self.g_hrs) if self.g_hrs > 0.0 else 0.5 * self.g_lrs

    @classmethod
    def from_measurements(  # pylint: disable=too-many-locals
        cls,
        hrs_samples: Sequence[float],
        lrs_samples: Sequence[float],
        *,
        g_converter: Optional[BinaryDeviceConductanceConverter] = None,
        distribution: str = "lognormal",
        retention: Optional[Dict[str, Tuple[Sequence[float], Sequence[float]]]] = None,
        **kwargs: Any,
    ) -> "TwoStateReRamNoiseModel":
        r"""Build the model from measured conductances (:math:`\mu S`).

        1. threshold :math:`g_{th} = \sqrt{\overline{hrs}\,\overline{lrs}}`;
        2. ``p_reset_fail`` is the fraction of HRS-targeted reads above
           :math:`g_{th}`, ``p_set_fail`` the fraction of LRS-targeted
           reads below;
        3. ``g_hrs`` / ``g_hrs_std`` are mean / std of the HRS reads below
           :math:`g_{th}`, ``g_lrs`` / ``g_lrs_std`` likewise above;
        4. with retention traces, ``nu_<state>_mean`` is the negative
           slope of :math:`\ln g` versus :math:`\ln((t + t_0)/t_0)`.

        Args:
            hrs_samples: conductances read after RESET pulses (all devices
                and cycles pooled).
            lrs_samples: conductances read after SET pulses.
            g_converter: unit cell; its ``g_lrs`` / ``g_hrs`` are replaced
                by the measured means.
            distribution: ``"lognormal"`` or ``"normal"``.
            retention: optional ``{"lrs": (t_s, g_mean), "hrs": (t_s,
                g_mean)}`` population-mean retention traces.
            kwargs: forwarded to the constructor.

        Returns:
            The fitted noise model.

        Raises:
            ValueError: if a population is empty or not separable.
        """
        hrs = np.asarray(hrs_samples, dtype=float)
        lrs = np.asarray(lrs_samples, dtype=float)
        if hrs.size == 0 or lrs.size == 0:
            raise ValueError("need at least one HRS and one LRS sample")

        threshold = sqrt(max(hrs.mean(), 1e-12) * lrs.mean())
        hrs_ok = hrs[hrs <= threshold]
        lrs_ok = lrs[lrs > threshold]
        if hrs_ok.size == 0 or lrs_ok.size == 0:
            raise ValueError("HRS and LRS populations are not separable")

        params: Dict[str, Any] = {
            "g_hrs": float(hrs_ok.mean()),
            "g_lrs": float(lrs_ok.mean()),
            "g_hrs_std": float(hrs_ok.std(ddof=1)) if hrs_ok.size > 1 else 0.0,
            "g_lrs_std": float(lrs_ok.std(ddof=1)) if lrs_ok.size > 1 else 0.0,
            "p_reset_fail": float(1.0 - hrs_ok.size / hrs.size),
            "p_set_fail": float(1.0 - lrs_ok.size / lrs.size),
            "distribution": distribution,
        }

        t_0 = float(kwargs.get("t_0", 20.0))
        if retention:
            for state in ("lrs", "hrs"):
                if state not in retention:
                    continue
                t_arr, g_arr = (np.asarray(a, dtype=float) for a in retention[state])
                mask = t_arr > 0
                x_log = np.log((t_arr[mask] + t_0) / t_0)
                y_log = np.log(g_arr[mask])
                slope = np.polyfit(x_log, y_log, 1)[0]
                params["nu_{}_mean".format(state)] = float(-slope)

        params.update(kwargs)
        if g_converter is not None:
            g_converter = deepcopy(g_converter)
            g_converter.g_lrs = g_converter.g_max = params["g_lrs"]
            g_converter.g_hrs = g_converter.g_min = params["g_hrs"]
        return cls(g_converter=g_converter, **params)

    def _is_lrs(self, g_values: Tensor) -> Tensor:
        """(P1) classify conductances into LRS (True) / HRS (False)."""
        return g_values > self.g_threshold

    @no_grad()
    def _sample_state(self, mean: float, std: float, like: Tensor) -> Tensor:
        """(P3) sample conductances of one state (mean / std in linear units)."""
        return sample_two_state_conductance(
            mean, std * self.prog_noise_scale, self.distribution, like
        )

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """(P1)-(P3): program every device to HRS or LRS."""
        target_lrs = self._is_lrs(g_target)

        flip = rand_like(g_target)
        set_failed = target_lrs & (flip < self.p_set_fail)
        reset_failed = (~target_lrs) & (flip < self.p_reset_fail)
        final_lrs = (target_lrs & ~set_failed) | reset_failed

        g_lrs = self._sample_state(self.g_lrs, self.g_lrs_std, g_target)
        g_hrs = self._sample_state(self.g_hrs, self.g_hrs_std, g_target)
        return where(final_lrs, g_lrs, g_hrs)

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """(R1) draw per-device exponents for both possible states.

        Returns:
            Tensor of shape ``[2, *g_target.shape]``; index 0 is used if
            the device ends up in LRS, index 1 if in HRS.
        """
        nu_lrs = self.nu_lrs_mean + self.nu_lrs_std * randn_like(g_target)
        nu_hrs = self.nu_hrs_mean + self.nu_hrs_std * randn_like(g_target)
        return stack([nu_lrs, nu_hrs]) * self.drift_scale

    @no_grad()
    def apply_drift_noise_to_conductance(  # pylint: disable=too-many-locals
        self, g_prog: Tensor, drift_noise_param: Optional[Tensor], t_inference: float
    ) -> Tensor:
        """(R1)-(R3): retention, state loss and read noise at ``t_inference``."""
        if drift_noise_param is None:
            drift_noise_param = self.generate_drift_coefficients(g_prog)
        is_lrs = self._is_lrs(g_prog)
        nu_drift = where(is_lrs, drift_noise_param[0], drift_noise_param[1])

        # (R1) power-law retention
        t_rel = (t_inference + self.t_0) / self.t_0
        g_drift = g_prog * (t_rel ** (-nu_drift)) if t_inference > 0 else g_prog.clone()

        # (R2) spontaneous state loss, probability per decade after t_0
        if t_inference > 0 and (self.p_retention_fail_lrs > 0 or self.p_retention_fail_hrs > 0):
            decades = log10(t_rel)
            p_lrs = 1.0 - (1.0 - self.p_retention_fail_lrs) ** decades
            p_hrs = 1.0 - (1.0 - self.p_retention_fail_hrs) ** decades
            lost = rand_like(g_prog) < where(
                is_lrs, g_prog.new_tensor(p_lrs), g_prog.new_tensor(p_hrs)
            )
            g_other = where(
                is_lrs,
                self._sample_state(self.g_hrs, self.g_hrs_std, g_prog),
                self._sample_state(self.g_lrs, self.g_lrs_std, g_prog),
            )
            g_drift = where(lost, g_other, g_drift)
            is_lrs = self._is_lrs(g_drift)

        # (R3) read noise, multiplicative, per state, with 1/f accumulation
        rel = where(
            is_lrs,
            g_prog.new_tensor(self.read_noise_rel_lrs),
            g_prog.new_tensor(self.read_noise_rel_hrs),
        )
        t_total = t_inference + self.t_0
        accum = sqrt(log((t_total + self.t_read) / (2.0 * self.t_read)))
        g_final = g_drift + g_drift * rel * accum * self.read_noise_scale * randn_like(g_drift)
        return g_final.clamp(min=0.0)

    @no_grad()
    def apply_programming_noise(self, weights: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Program a weight matrix (called once by ``program_analog_weights``).

        Returns:
            ``(programmed_weights, drift_params)`` where ``drift_params``
            holds one tensor ``[g_prog, nu_lrs, nu_hrs, scale_ratio,
            cell_mask]`` of shape ``[5, *weights.shape]`` per device slice.
        """
        target_conductances, params = self.g_converter.convert_to_conductances(weights)
        scale = (
            params["scale_ratio"].to(weights)
            if isinstance(params["scale_ratio"], Tensor)
            else weights.new_tensor(params["scale_ratio"])
        )
        cell_masks = params.get("cell_mask", [None] * len(target_conductances))

        programmed = []
        drift_params = []
        for g_target, mask in zip(target_conductances, cell_masks):
            g_prog = self.apply_programming_noise_to_conductance(g_target)
            if mask is not None:  # reference entries are not devices: keep them exact
                g_prog = where(mask, g_prog, g_target)
                mask_row = mask.to(g_prog.dtype)
            else:
                mask_row = g_prog.new_ones(g_prog.shape)
            nu_both = self.generate_drift_coefficients(g_target)
            programmed.append(g_prog)
            drift_params.append(
                stack([g_prog, nu_both[0], nu_both[1], scale.expand_as(g_prog), mask_row])
            )

        return self.g_converter.convert_back_to_weights(programmed, params), drift_params

    @no_grad()
    def apply_drift_noise(
        self, weights: Tensor, drift_noise_parameters: List[Optional[Tensor]], t_inference: float
    ) -> Tensor:
        """Drift the *stored* programmed conductances (no re-quantisation).

        ``weights`` is only used when no programming information is
        available; the weights are then programmed first.
        """
        if drift_noise_parameters is None or any(p is None for p in drift_noise_parameters):
            weights, programmed_params = self.apply_programming_noise(weights)
        else:
            programmed_params = [p for p in drift_noise_parameters if p is not None]

        converter = cast(BinaryDeviceConductanceConverter, self.g_converter)
        params = {
            "scale_ratio": programmed_params[0][3].flatten()[0],
            "f_lst": converter.f_lst,
            "g_lrs": converter.g_lrs,
            "g_hrs": converter.g_hrs,
        }
        drifted = []
        for prog in programmed_params:
            g_drift = self.apply_drift_noise_to_conductance(prog[0], prog[1:3], t_inference)
            if prog.shape[0] > 4:  # reference entries do not drift
                g_drift = where(prog[4] > 0.5, g_drift, prog[0])
            drifted.append(g_drift)
        return converter.convert_back_to_weights(drifted, params)

    @no_grad()
    def apply_noise(self, weights: Tensor, t_inference: float) -> Tensor:
        """Program and drift in one shot (fresh samples every call)."""
        programmed, drift_params = self.apply_programming_noise(weights)
        return self.apply_drift_noise(programmed, drift_params, t_inference)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """Constructor keyword arguments (without the converter), JSON-serialisable."""
        keys = [
            "g_lrs",
            "g_hrs",
            "g_lrs_std",
            "g_hrs_std",
            "distribution",
            "p_set_fail",
            "p_reset_fail",
            "prog_noise_scale",
            "nu_lrs_mean",
            "nu_lrs_std",
            "nu_hrs_mean",
            "nu_hrs_std",
            "p_retention_fail_lrs",
            "p_retention_fail_hrs",
            "drift_scale",
            "t_0",
            "read_noise_rel_lrs",
            "read_noise_rel_hrs",
            "read_noise_scale",
            "t_read",
        ]
        return {key: getattr(self, key) for key in keys}
