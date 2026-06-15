"""
Testing suite for electron distribution functions.

This module provides unit testing for the electron distribution functions in
:py:mod:`trilobite.radiation.synchrotron.electron_distributions`. The tests ensure that the
electron distribution functions behave as expected under various conditions, including
normalization, energy limits, and consistency with theoretical predictions.
"""

import pytest
import numpy as np
from scipy.integrate import quad
from trilobite.radiation.synchrotron.electron_distributions import (
    ElectronDistribution,
    PowerLaw,
    BrokenPowerLaw,
    MaxwellJuettner,
    MaxwellJuettnerPowerLaw,
    MaxwellJuettnerBrokenPowerLaw,
)


class BaseTestElectronDistribution:
    """
    Base test class for electron distributions.

    This class provides common testing methods for all electron distribution functions. Each specific
    distribution test class should inherit from this base class and define the necessary parameters
    and expected results.
    """

    DISTRIBUTION_CLASS: ElectronDistribution = None  # To be defined in subclasses
    GAMMA_ARRAY: np.ndarray = None

    PARAMETER_SETS: list[dict] = []
    EXPECTED_SUPPORTS: list[tuple] = []

    # ------------------------------------------- #
    # Structural Testing                          #
    # ------------------------------------------- #
    # Structural testing ensures that the relevant electron distribution class is
    # properly defined, callable with the correct inputs, etc. This is the basic sanity
    # check before we do any correctness testing.
    def test_support(self):
        """Test that the support of the distribution is as expected."""
        for params, expected_support in zip(self.PARAMETER_SETS, self.EXPECTED_SUPPORTS):
            # Compute the support
            computed_support = self.DISTRIBUTION_CLASS.support(**params)

            # Assert that the computed support matches the expected support
            assert computed_support == expected_support, (
                f"Support mismatch for parameters {params}: expected {expected_support}, got {computed_support}"
            )

    def test_pdf_callable(self):
        """Test that the pdf gives relevant values for a range of parameters."""
        for params in self.PARAMETER_SETS:
            # Compute the support and the PDF.
            support = self.DISTRIBUTION_CLASS.support(**params)
            pdf_values = self.DISTRIBUTION_CLASS.pdf(self.GAMMA_ARRAY, **params)

            # Ensure that the PDF is zero outside the support.
            assert np.all(pdf_values[self.GAMMA_ARRAY < support[0]] == 0), (
                f"PDF is non-zero below support for parameters {params}"
            )

            # Ensure that the PDF is positive within the support.
            assert np.all(pdf_values[(self.GAMMA_ARRAY >= support[0]) & (self.GAMMA_ARRAY <= support[1])] >= 0), (
                f"PDF is negative within support for parameters {params}"
            )

    def test_pdf_scalar_callable(self):
        """Test that the PDF supports scalar gamma inputs."""
        gamma = float(self.GAMMA_ARRAY[len(self.GAMMA_ARRAY) // 2])

        for params in self.PARAMETER_SETS:
            pdf_value = self.DISTRIBUTION_CLASS.pdf(gamma, **params)

            assert np.ndim(pdf_value) == 0, (
                f"Scalar PDF call did not return a scalar-like value for parameters {params}."
            )

    def test_norm_scales_pdf(self):
        """Test that the ``norm`` argument scales the PDF linearly."""
        norm = 3.7

        for params in self.PARAMETER_SETS:
            pdf_unit = self.DISTRIBUTION_CLASS.pdf(
                self.GAMMA_ARRAY,
                norm=1.0,
                **params,
            )
            pdf_scaled = self.DISTRIBUTION_CLASS.pdf(
                self.GAMMA_ARRAY,
                norm=norm,
                **params,
            )

            assert np.allclose(pdf_scaled, norm * pdf_unit), (
                f"`norm` does not linearly scale PDF for parameters {params}."
            )

    # ------------------------------------------- #
    # Moment / Derived Quantity Smoke Tests       #
    # ------------------------------------------- #
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_moment_callable(self, order):
        """Test that moments used by the base class are callable."""
        for params in self.PARAMETER_SETS:
            moment = self.DISTRIBUTION_CLASS.moment(order, **params)

            assert np.ndim(moment) == 0, f"Moment order {order} did not return a scalar for parameters {params}."

            assert not np.isnan(moment), f"Moment order {order} returned NaN for parameters {params}."

    def test_n_total_callable(self):
        """Test that the total number method is callable."""
        for params in self.PARAMETER_SETS:
            n_total = self.DISTRIBUTION_CLASS.n_total(norm=1.0, **params)

            assert np.ndim(n_total) == 0
            assert not np.isnan(n_total)

    def test_n_eff_callable(self):
        """Test that the effective radiating number method is callable."""
        for params in self.PARAMETER_SETS:
            n_eff = self.DISTRIBUTION_CLASS.n_eff(norm=1.0, **params)

            assert np.ndim(n_eff) == 0
            assert not np.isnan(n_eff)

    def test_mean_callable(self):
        """Test that the mean Lorentz factor is callable."""
        for params in self.PARAMETER_SETS:
            mean = self.DISTRIBUTION_CLASS.mean(**params)

            assert np.ndim(mean) == 0
            assert not np.isnan(mean)

    def test_variance_callable(self):
        """Test that the variance is callable."""
        for params in self.PARAMETER_SETS:
            variance = self.DISTRIBUTION_CLASS.var(**params)

            assert np.ndim(variance) == 0
            assert not np.isnan(variance)

    def test_standard_deviation_callable(self):
        """Test that the standard deviation is callable."""
        for params in self.PARAMETER_SETS:
            std = self.DISTRIBUTION_CLASS.std(**params)

            assert np.ndim(std) == 0
            assert not np.isnan(std)

    # ------------------------------------------- #
    # Numerical Consistency Testing               #
    # ------------------------------------------- #
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_moment_matches_quadrature(self, order):
        """Test that analytic moments match direct numerical quadrature."""
        for params in self.PARAMETER_SETS:
            gamma_min, gamma_max = self.DISTRIBUTION_CLASS.support(**params)

            quadrature_moment, _ = quad(
                lambda gamma: (
                    gamma**order
                    * self.DISTRIBUTION_CLASS.pdf(
                        gamma,
                        norm=1.0,
                        **params,
                    )
                ),
                gamma_min,
                gamma_max,
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=200,
            )

            analytic_moment = self.DISTRIBUTION_CLASS.moment(
                order,
                **params,
            )

            assert analytic_moment == pytest.approx(
                quadrature_moment,
                rel=1.0e-7,
                abs=1.0e-10,
            ), (
                f"Moment order {order} does not match quadrature for "
                f"parameters {params}: expected {quadrature_moment}, "
                f"got {analytic_moment}"
            )


class TestPowerLaw(BaseTestElectronDistribution):
    DISTRIBUTION_CLASS = PowerLaw
    GAMMA_ARRAY = np.linspace(1, 1e6, 1000)

    PARAMETER_SETS = [
        {"p": 2.5, "gamma_min": 1, "gamma_max": 1e6},
        {"p": 3.0, "gamma_min": 10, "gamma_max": 1e5},
        {"p": 2.0, "gamma_min": 1, "gamma_max": 1e4},
    ]
    EXPECTED_SUPPORTS = [
        (1, 1e6),
        (10, 1e5),
        (1, 1e4),
    ]


class TestBrokenPowerLaw(BaseTestElectronDistribution):
    DISTRIBUTION_CLASS = BrokenPowerLaw
    GAMMA_ARRAY = np.linspace(1, 1e6, 1000)

    PARAMETER_SETS = [
        {"p1": 2.0, "p2": 3.0, "gamma_c": 100, "gamma_min": 1, "gamma_max": 1e6},
        {"p1": 2.5, "p2": 3.5, "gamma_c": 500, "gamma_min": 10, "gamma_max": 1e5},
        {"p1": 1.5, "p2": 2.5, "gamma_c": 50, "gamma_min": 1, "gamma_max": 1e4},
    ]
    EXPECTED_SUPPORTS = [
        (1, 1e6),
        (10, 1e5),
        (1, 1e4),
    ]

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_moment_matches_quadrature(self, order):
        """Test that analytic moments match direct numerical quadrature.

        We override the base class test to use a specified break point in the quadrature because
        the discontinuity would otherwise cause the quadrature to be significantly inaccurate.
        """
        for params in self.PARAMETER_SETS:
            gamma_min, gamma_max = self.DISTRIBUTION_CLASS.support(**params)

            quadrature_moment, _ = quad(
                lambda gamma: (
                    gamma**order
                    * self.DISTRIBUTION_CLASS.pdf(
                        gamma,
                        norm=1.0,
                        **params,
                    )
                ),
                gamma_min,
                gamma_max,
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=200,
                points=[params["gamma_c"]],
            )

            analytic_moment = self.DISTRIBUTION_CLASS.moment(
                order,
                **params,
            )

            assert analytic_moment == pytest.approx(
                quadrature_moment,
                rel=1.0e-7,
                abs=1.0e-10,
            ), (
                f"Moment order {order} does not match quadrature for "
                f"parameters {params}: expected {quadrature_moment}, "
                f"got {analytic_moment}"
            )


class TestMaxwellJuettner(BaseTestElectronDistribution):
    DISTRIBUTION_CLASS = MaxwellJuettner
    GAMMA_ARRAY = np.linspace(1, 1e3, 1000)

    PARAMETER_SETS = [
        {"Theta": 0.1},
        {"Theta": 1.0},
        {"Theta": 10.0},
    ]
    EXPECTED_SUPPORTS = [
        (1, np.inf),
        (1, np.inf),
        (1, np.inf),
    ]

    def test_approx_first_moment_matches_kinetic_plus_rest_mass(self):
        """Test that the approximate total moment is kinetic plus rest mass."""
        for params in self.PARAMETER_SETS:
            first_moment = self.DISTRIBUTION_CLASS.approx_first_moment(**params)
            kinetic_moment = self.DISTRIBUTION_CLASS.approx_first_kinetic_moment(**params)

            assert first_moment == pytest.approx(kinetic_moment + 1.0), (
                f"Approximate first moment is not kinetic moment plus one for parameters {params}."
            )

    def test_approx_first_moment_is_close_to_exact_first_moment(self):
        """Test that the approximate first moment is close to the exact first moment."""
        for params in self.PARAMETER_SETS:
            exact_moment = self.DISTRIBUTION_CLASS.moment(1, **params)
            approx_moment = self.DISTRIBUTION_CLASS.approx_first_moment(**params)

            assert approx_moment == pytest.approx(exact_moment, rel=5.0e-2), (
                f"Approximate first moment is not close to exact first moment "
                f"for parameters {params}: expected {exact_moment}, "
                f"got {approx_moment}."
            )


class TestMaxwellJuettnerPowerLaw(BaseTestElectronDistribution):
    DISTRIBUTION_CLASS = MaxwellJuettnerPowerLaw
    GAMMA_ARRAY = np.linspace(1, 1e6, 1000)

    PARAMETER_SETS = [
        {"Theta": 0.1, "p": 2.5, "gamma_min": 1, "gamma_max": 1e6, "delta": 1.0},
        {"Theta": 1.0, "p": 3.0, "gamma_min": 10, "gamma_max": 1e5, "delta": 1.0},
        {"Theta": 10.0, "p": 2.0, "gamma_min": 1, "gamma_max": 1e4, "delta": 1.0},
        {"Theta": 0.1, "p": 2.5, "gamma_min": 1, "gamma_max": 1e6, "delta": 0.5},
        {"Theta": 1.0, "p": 3.0, "gamma_min": 10, "gamma_max": 1e5, "delta": 0.5},
        {"Theta": 10.0, "p": 2.0, "gamma_min": 1, "gamma_max": 1e4, "delta": 0.5},
    ]
    EXPECTED_SUPPORTS = [
        (1, np.inf),
        (1, np.inf),
        (1, np.inf),
        (1, np.inf),
        (1, np.inf),
        (1, np.inf),
    ]

    @pytest.mark.skip(reason="No easy quadrature check + composition")
    def test_moment_matches_quadrature(self, order):
        """Test that analytic moments match direct numerical quadrature."""
        pass

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_composition_quadrature(self, order):
        """Test mixed moments by quadraturing each component separately."""
        for params in self.PARAMETER_SETS:
            delta = params["delta"]

            mj_params = {
                "Theta": params["Theta"],
            }
            pl_params = {
                "p": params["p"],
                "gamma_min": params["gamma_min"],
                "gamma_max": params["gamma_max"],
            }

            mj_moment_quad, _ = quad(
                lambda gamma: (
                    gamma**order
                    * MaxwellJuettner.pdf(
                        gamma,
                        norm=1.0,
                        **mj_params,
                    )
                ),
                1.0,
                np.inf,
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=500,
            )

            pl_moment_quad, _ = quad(
                lambda gamma: (
                    gamma**order
                    * PowerLaw.pdf(
                        gamma,
                        norm=1.0,
                        **pl_params,
                    )
                ),
                pl_params["gamma_min"],
                pl_params["gamma_max"],
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=500,
            )

            pl_m0_quad, _ = quad(
                lambda gamma: PowerLaw.pdf(
                    gamma,
                    norm=1.0,
                    **pl_params,
                ),
                pl_params["gamma_min"],
                pl_params["gamma_max"],
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=500,
            )

            expected_moment = delta * mj_moment_quad + (1.0 - delta) * pl_moment_quad / pl_m0_quad

            computed_moment = self.DISTRIBUTION_CLASS.moment(
                order,
                **params,
            )

            assert computed_moment == pytest.approx(
                expected_moment,
                rel=1.0e-6,
                abs=1.0e-10,
            ), (
                f"Mixed moment order {order} does not match component-wise "
                f"quadrature for parameters {params}: expected "
                f"{expected_moment}, got {computed_moment}."
            )


class TestMaxwellJuettnerBrokenPowerLaw(BaseTestElectronDistribution):
    DISTRIBUTION_CLASS = MaxwellJuettnerBrokenPowerLaw
    GAMMA_ARRAY = np.linspace(1, 1e6, 1000)

    PARAMETER_SETS = [
        {"Theta": 0.1, "p1": 2.0, "p2": 3.0, "gamma_c": 100, "delta": 0.5, "gamma_min": 1, "gamma_max": 1e6},
        {"Theta": 1.0, "p1": 2.5, "p2": 3.5, "gamma_c": 500, "delta": 0.5, "gamma_min": 10, "gamma_max": 1e5},
        {"Theta": 10.0, "p1": 1.5, "p2": 2.5, "gamma_c": 50, "delta": 0.5, "gamma_min": 1, "gamma_max": 1e4},
    ]
    EXPECTED_SUPPORTS = [
        (1, np.inf),
        (1, np.inf),
        (1, np.inf),
    ]

    @pytest.mark.skip(reason="No easy quadrature check + composition")
    def test_moment_matches_quadrature(self, order):
        """Test that analytic moments match direct numerical quadrature."""
        pass

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_composition_quadrature(self, order):
        """Test mixed moments by quadraturing each component separately."""
        for params in self.PARAMETER_SETS:
            delta = params["delta"]

            mj_params = {
                "Theta": params["Theta"],
            }
            bpl_params = {
                "p1": params["p1"],
                "p2": params["p2"],
                "gamma_c": params["gamma_c"],
                "gamma_min": params["gamma_min"],
                "gamma_max": params["gamma_max"],
            }

            mj_moment_quad, _ = quad(
                lambda gamma: (
                    gamma**order
                    * MaxwellJuettner.pdf(
                        gamma,
                        norm=1.0,
                        **mj_params,
                    )
                ),
                1.0,
                np.inf,
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=500,
            )

            pl_moment_quad, _ = quad(
                lambda gamma: (
                    gamma**order
                    * BrokenPowerLaw.pdf(
                        gamma,
                        norm=1.0,
                        **bpl_params,
                    )
                ),
                bpl_params["gamma_min"],
                bpl_params["gamma_max"],
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=500,
                points=[params["gamma_c"]],
            )

            pl_m0_quad, _ = quad(
                lambda gamma: BrokenPowerLaw.pdf(gamma, norm=1.0, **bpl_params),
                bpl_params["gamma_min"],
                bpl_params["gamma_max"],
                epsabs=1.0e-10,
                epsrel=1.0e-8,
                limit=500,
                points=[params["gamma_c"]],
            )

            expected_moment = delta * mj_moment_quad + (1.0 - delta) * pl_moment_quad / pl_m0_quad

            computed_moment = self.DISTRIBUTION_CLASS.moment(
                order,
                **params,
            )

            assert computed_moment == pytest.approx(
                expected_moment,
                rel=1.0e-6,
                abs=1.0e-10,
            ), (
                f"Mixed moment order {order} does not match component-wise "
                f"quadrature for parameters {params}: expected "
                f"{expected_moment}, got {computed_moment}."
            )
