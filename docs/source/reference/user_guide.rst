.. _user_guide:
========================
Triceratops User Guide
========================
Welcome to the **Triceratops User Guide**! This page is the location of all technical documentation for the
triceratops library and is generally the first place to look when seeking information on how to use the library. In
addition to the resources on this page, the API reference (:ref:`api`) gives in-depth information about call signatures
and code structure. The developer guide (:ref:`developer_guide`) is a resource for those who want to contribute to the library or
understand the codebase in detail.

Topics in this guide are broken down by subject area, with each section providing
a comprehensive overview of the relevant components and their usage.

Getting Started
---------------

To get started with triceratops, check out the :ref:`getting_started` guide which walks through installation,
basic usage, and a simple example. This is a great place to begin if you're new to triceratops or scientific
Python in general. You can also explore the :ref:`examples` section for more in-depth tutorials and use cases.

Data Loading, Handling, and Visualization
-----------------------------------------

The first step in any radio analysis is to load and visualize your data. The data modules in triceratops provide
tools for loading, processing, and visualizing observational data. Most importantly, these structures are the entry
point to the library, providing a consistent interface for working with different types of data in our model and
inference pipelines. The following guides cover the relevant functionality:

.. toctree::
   :maxdepth: 2

   data/overview


Triceratops Models
-------------


Building Blocks
---------------

Behind every Triceratops model are a set of modular building blocks that define the physical processes and components
of the system being modeled. These building blocks can be combined in various ways to create complex models that
capture the nuances of radio observations. The following guides provide an overview of the various modules and
the constituent physics:



Extensions and 3rd Party Libraries
----------------------------------


Configuration and Setup
------------------------


Methodology, Theory, and Systems
---------------------------------
As a highly technical library, triceratops relies on a number of advanced methodologies and theoretical concepts to
achieve its goals. This section provides an overview of the key methodologies and theories that underpin the
library, including the mathematical foundations and algorithms used in the modeling and inference processes. Where
possible, we cover high level details of relevant theory in the broad theory-based sections and provide
more detailed, case-specific, considerations to the discussions about specific systems.


Systems
^^^^^^^

Below are a number of guides that cover specific systems modeled in triceratops, including the relevant physics,
methodologies, and considerations for each system.

Supernovae
~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   physics/supernovae/shocks
