from collections import namedtuple

import pytest
from astropy import units as u

from triceratops.models.core import Model, ModelParameter, ModelVariable

# ============================================================
# Helpers
# ============================================================

# Reusable namedtuple definitions
SimpleOutputs = namedtuple("SimpleOutputs", ["y"])


# ============================================================
# Minimal Valid Model
# ============================================================


class SimpleModel(Model):
    """
    Minimal valid concrete model for base-class testing.
    """

    PARAMETERS = (
        ModelParameter(
            name="a",
            default=1.0,
            base_units=u.dimensionless_unscaled,
        ),
    )

    VARIABLES = (
        ModelVariable(
            name="x",
            base_units=u.dimensionless_unscaled,
        ),
    )

    OUTPUTS = SimpleOutputs
    UNITS = SimpleOutputs(y=u.dimensionless_unscaled)

    def _forward_model(self, variables, parameters):
        return {"y": parameters["a"] * variables["x"]}


# ============================================================
# Subclass Validation Tests
# ============================================================


def test_model_requires_variables():
    """Model subclasses must declare at least one VARIABLE."""

    with pytest.raises(ValueError):

        class _BadModel(Model):
            PARAMETERS = ()
            VARIABLES = ()

            Outputs = namedtuple("BadOutputs", ["y"])
            Units = namedtuple("BadUnits", ["y"])

            OUTPUTS = Outputs(y="y")
            UNITS = Units(y=u.dimensionless_unscaled)

            def _forward_model(self, variables, parameters):
                return {"y": 0.0}


def test_outputs_units_field_mismatch():
    """OUTPUTS and UNITS must have identical field names."""

    with pytest.raises(TypeError):

        class _BadModel(Model):
            PARAMETERS = (ModelParameter("a", 1.0, base_units=u.dimensionless_unscaled),)

            VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

            OUTPUTS = SimpleOutputs
            UNITS = SimpleOutputs(y=u.dimensionless_unscaled, z=u.dimensionless_unscaled)

            def _forward_model(self, variables, parameters):
                return {"y": 0.0}


# ============================================================
# Basic Forward Model Behavior
# ============================================================


def test_forward_model_default_parameter():
    """Missing parameters should be filled with defaults."""

    model = SimpleModel()
    result = model({"x": 2.0}, {})

    assert result["y"].value == 2.0
    assert result["y"].unit == u.dimensionless_unscaled


def test_forward_model_parameter_override():
    """Provided parameters should override defaults."""

    model = SimpleModel()
    result = model({"x": 2.0}, {"a": 3.0})

    assert result["y"].value == 6.0


def test_forward_model_missing_variable_raises():
    """Missing required variable should raise ValueError."""

    model = SimpleModel()

    with pytest.raises(ValueError):
        model({}, {})


# ============================================================
# Bounds Checking
# ============================================================


def test_parameter_bounds_violation():
    """Parameter bounds should be enforced."""

    # Define bounded model
    class BoundedModel(SimpleModel):
        PARAMETERS = (
            ModelParameter(
                name="a",
                default=1.0,
                base_units=u.dimensionless_unscaled,
                bounds=(0.0, 10.0),
            ),
        )

    model = BoundedModel()

    with pytest.raises(ValueError):
        model({"x": 1.0}, {"a": -1.0})


# ============================================================
# Metadata Properties
# ============================================================


def test_metadata_properties():
    """parameter_names, variable_names, and output_names should be correct."""

    model = SimpleModel()

    assert model.parameter_names == ("a",)
    assert model.variable_names == ("x",)
    assert model.output_names == ("y",)


# ============================================================
# Tupled Output Order
# ============================================================


def test_forward_model_tupled():
    """_forward_model_tupled should respect OUTPUT ordering."""

    model = SimpleModel()

    result = model._forward_model_tupled(
        {"x": 2.0},
        {"a": 3.0},
    )

    assert isinstance(result, tuple)
    assert result[0] == 6.0
