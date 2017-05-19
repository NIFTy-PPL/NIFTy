Spaces
======
The :py:class:`Space` classes of NIFTY represent geometrical spaces approximated by grids in the computer environment. Each subclass of the base class corresponds to a specific grid type and replaces some of the inherited methods with its own methods that are unique to the respective grid. This framework ensures an abstract handling of spaces independent of the underlying geometrical grid and the grid's resolution.

Each instance of a :py:class:`Space` needs to capture all structural and dimensional specifics of the grid and all computationally relevant quantities such as the data type of associated field values. These parameters are stored as properties of an instance of the class at its initialization, and they do not need to be accessed explicitly by the user thereafter. This prevents the writing of grid or resolution dependent code.

Spatial symmetries of a system can be exploited by corresponding coordinate transformations. Often, transformations from one basis to its harmonic counterpart can greatly reduce the computational complexity of algorithms. The harmonic basis is defined by the eigenbasis of the Laplace operator; e.g., for a flat position space it is the Fourier basis. This conjugation of bases is implemented in NIFTY by distinguishing conjugate space classes, which can be obtained by the instance method *get_codomain* (and checked for by *check_codomain*). Moreover, transformations between conjugate spaces are performed automatically if required.


Space classes
-------------
Next to the generic :py:class:`Space` class, NIFTY has implementations of five subclasses, representing specific geometrical spaces and their discretizations.

.. toctree::
    :maxdepth: 1

    rg_space
    hp_space
    gl_space
    lm_space
    power_space
    power_indices

.. currentmodule:: nifty

The ``Space`` class -- The base Space object
--------------------------------------------

.. autoclass:: Space
    :show-inheritance:
    :members:
