.. _inference:
========================================
Parameter Inference and Model Comparison
========================================

- Triceratops designed for plugging the models from :mod:`models` into inference pipelines.
- parameter estimation, MCMC, nested sampling, model comparison.
- inference modules provide tools for setting up and running inference analyses using various sampling algorithms.
- integrates with third-party libraries like :mod:`emcee`, :mod:`dynesty`, and :mod:`bilby`.
- supports custom likelihood functions and priors.
- flexible configuration options for tailoring inference analyses to specific scientific goals.

The Triceratops Inference Pipeline
-----------------------------------

.. graphviz:: ../../images/inference/inference_diagram.dot

----

As with any robust statistical pipeline, there are a lot of options and configurations that make the inference
pipeline look more complicated than it actually is. Before introducing any of the specifics about inference,
its worth taking a step back and looking at the overall structure of the inference pipeline in Triceratops.
The diagram above provides a high-level overview of the inference pipeline in Triceratops. We'll discuss them
step by step:

1. **The Model**: The first step in any inference pipeline in Triceratops is the model. This can either be
   a model that's already built into the :mod:`models` module, or it can be a custom model. In effect, this provides
   a mapping from some input variables :math:`{\bf x}` and some set of parameters :math:`\boldsymbol{\Theta}` to
   some set of observables :math:`{\bf y}`:

   .. math::

         {\bf y} = \mathcal{M}({\bf x}; \boldsymbol{\Theta})

   The parameters of each model are defined by the model class.

2. **The Dataset**: With the model in hand, we also need the data we want to fit too. The idea is to find the
   parameters of the model which best predict the dataset. Thus, the dataset provides a set of observed
   :math:`{\bf x}_{\rm obs}` and :math:`{\bf y}_{\rm obs}` values. These are typically loaded using the
   :mod:`data` module.

3. **The Likelihood Function**: The likelihood function quantifies how well the model predictions match
   the observed data for a given set of parameters. In Triceratops, these are :class:`inference.likelihood.base.Likelihood`
   objects, which provide a function :math:`\mathcal{L}(\boldsymbol{\Theta} | {\bf x}_{\rm obs}, {\bf y}_{\rm obs})` that
   computes the likelihood of the observed data given the model parameters. Triceratops provides a variety of
   built-in likelihood functions, and users can also define custom likelihoods as needed.

4. **Inference Problems**: The likelihood alone isn't enough to completely specify an inference problem. We also need
   to provide priors on the model parameters. In Triceratops, an inference problem is defined by a combination of a
   model, a dataset, a likelihood function, and a set of priors. This is encapsulated in the
   :class:`~inference.problem.InferenceProblem`.

5. **Samplers**: With the inference problem defined, we can now use a sampling algorithm to explore the parameter
   space and estimate the posterior distribution of the model parameters. Triceratops integrates with several
   third-party sampling libraries, including :mod:`emcee`, :mod:`dynesty`, and :mod:`bilby`. Each of these
   samplers has its own strengths and weaknesses, and Triceratops provides a unified interface for using them through the
   :class:`~inference.sampling.base.Sampler` class and its subclasses.

6. **Results**: After running the sampler, we obtain a set of samples from the posterior distribution of the
   model parameters. These samples can be analyzed to compute summary statistics, generate plots, and perform
   model comparison. Triceratops provides tools for working with the results of inference analyses, which are specific
   to each sampler used.

This is the standard workflow for performing inference in Triceratops. The following sections will provide more details
on each of these components, along with examples of how to use them in practice.

----

Likelihoods
------------

The Likelihood Class
^^^^^^^^^^^^^^^^^^^^

In Triceratops, **likelihood functions** quantify how well a physical model reproduces a
dataset under a particular statistical noise model. Conceptually, a likelihood defines

.. math::

    \mathcal{L}(\boldsymbol{\Theta}\mid \mathcal{D}),

the probability of observing the dataset :math:`\mathcal{D}` given model parameters
:math:`\boldsymbol{\Theta}`.

Rather than treating likelihoods as opaque black boxes, Triceratops implements them as
**structured objects** that explicitly bind together:

- a **model** (from :mod:`models`),
- a **data container** (from :mod:`data`),
- and a **noise/statistical assumption** (implemented by the likelihood subclass).

All likelihoods inherit from :class:`~inference.likelihood.base.Likelihood`.

.. currentmodule:: inference.likelihood.base

Creating a Likelihood Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A likelihood object is constructed by combining a model instance with a data container.
In most workflows, you will do this once, and then reuse the likelihood many times during
sampling.

A typical initialization pattern looks like:

.. code-block:: python

    from triceratops.models import MyModel
    from triceratops.data import RadioLightCurveContainer
    from triceratops.inference.likelihood import MyLikelihood

    model = MyModel(...)
    data = RadioLightCurveContainer.from_file(
        "my_lightcurve.ecsv",
        frequency=6.0,   # GHz if unitless
        band="VLA C-band"
    )

    like = MyLikelihood(model=model, data=data)

During initialization, likelihoods typically do three things:

1. **Compatibility validation**
   Likelihoods enforce that the provided model/data pairing is meaningful (e.g., a time-domain
   likelihood should not accept a heterogeneous-frequency dataset). This failure mode is
   intentionally *eager*: it is better to raise during construction than after thousands of
   sampler calls.

2. **Dataset retention (for provenance)**
   The original :class:`astropy.table.Table` and/or container object is retained internally so
   that the likelihood can be inspected, serialized, or re-plotted later.

3. **Preprocessing for performance**
   Repeated operations like unit conversion, sorting, masking detections vs. upper limits, and
   building dense numerical arrays are performed once via :meth:`Likelihood._process_input_data`.
   This prepares cached arrays that the optimized backend can consume with minimal overhead.

The Likelihood Function
~~~~~~~~~~~~~~~~~~~~~~~

Likelihood evaluation is split into two layers: a public wrapper and an optimized backend.

.. tab-set::

    .. tab-item:: High-Level API

        The high-level interface is :meth:`Likelihood.log_likelihood`. This method is designed
        to be called by user code. It accepts model parameters as kwargs, performs coercion of unit-bearing
        quantities, and validates that inputs are well-formed.

        .. code-block:: python

            # Example parameterization (exact signature depends on your model / likelihood)
            lnL = like.log_likelihood(
                theta_E=0.12,
                epsilon_B=1e-2,
                n0=0.5,
            )

        This is generally the nice way to call the likelihood if you don't need to do so many many times. If
        you are making a call to the likelihood inside of a tight loop (e.g., inside a sampler), you may want to
        consider using the low-level API instead.

    .. tab-item:: Low-Level API

        The low-level backend is :meth:`Likelihood._log_likelihood`. This method should be written
        for performance: it assumes all inputs are already validated and coerced into the expected
        internal representation (e.g., floats/NumPy arrays in a consistent unit system).

        This method is where you should implement the actual statistical model (Gaussian errors,
        censored likelihood for upper limits, correlated noise, etc.).

        .. code-block:: python

            # Inside a Likelihood subclass:
            def _log_likelihood(self, **pars) -> float:
                # assumes pars already validated and in internal form
                mu = self._model_prediction(**pars)
                resid = (self._y - mu) / self._sigma
                return -0.5 * (resid**2).sum()

While it is not common to need to implement a custom likelihood, we recognize that some users
may wish / need to do so. For that purpose, see the separate documentation on building likelihood
functions: :ref:`likelihood_development`.

Existing Likelihoods
^^^^^^^^^^^^^^^^^^^^

In most cases, a likelihood function has already been implemented for your use case / data type. Below are
the various likelihoods that are currently available in Triceratops. For more details on each likelihood,
see the corresponding documentation pages.

.. currentmodule:: triceratops.inference.likelihood

.. autosummary::
    :toctree:

    base.Likelihood
    single_epoch_photometry.GaussianPhotometryLikelihood

----

Inference Problems
------------------

.. hint::

    For more information about inference problem development, implementation, and parameter management,
    see :ref:`inference_problem_dev`.

Once you have a model, data, and likelihood function, you can combine them into an
**inference problem**. An inference problem bundles together all the ingredients needed to
perform parameter estimation or model comparison.

The Parts of an Inference Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The model
- The likelihood
- The data
- The parameters, priors, initial values, bounds, etc.

point to them all being defined in the :class:`~inference.problem.InferenceProblem` class.

The InferenceProblem Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduce the class as the container for everything.

Initializing an Inference Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Simply provide a likelihood to the inference problem object.
- There may be additional kwargs if needed, but not usually.

Working with Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

    For more information about priors, including custom priors, see :ref:`priors_dev`.

- How to access the parameters and see their properties
- Setting initial values
- Setting priors
- Setting bounds
- freezing and free parameters
- etc.

Computing the Prior and Posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- How to compute the log likelihood, log prior, and log posterior for a given set of parameters.
- How to view the initial values of these various parameters
- etc.

Validating for Inference
~~~~~~~~~~~~~~~~~~~~~~~~~

Samplers
--------

.. hint::

    For more information about sampler development, implementation, and integration, see :ref:`sampler_dev`.

What is a sampler: a way to explore the parameter space and find the minimum of the posterior distribution.
Triceratops provides interfaces to several popular sampling libraries, including :mod:`emcee`, :mod:`dynesty`, and :mod:`bilby`.
