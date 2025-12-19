"""Plotting utility functions."""

import matplotlib.pyplot as plt


def resolve_fig_axes(fig=None, axes=None, fig_size=(8, 6)):
    """
    Resolve and return a figure and axes for plotting.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        An existing figure object. If None, a new figure will be created.
    axes : matplotlib.axes.Axes, optional
        An existing axes object. If None, axes will be created or retrieved from the figure.
    fig_size : tuple, optional
        Size of the figure to create if fig is None. Default is (8, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resolved figure object.
    """
    import matplotlib.pyplot as plt

    if fig is None and axes is None:
        fig, axes = plt.subplots(figsize=fig_size)
    elif fig is not None and axes is None:
        axes = fig.gca()
    elif fig is None and axes is not None:
        fig = axes.figure

    return fig, axes


def set_plot_style():
    """Set the global plot style for matplotlib figures."""
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{graphicx,amsmath,amssymb,amsfonts,algorithmicx,algorithm,algpseudocodex}"
    )
    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["xtick.minor.size"] = 5
    plt.rcParams["ytick.major.size"] = 8
    plt.rcParams["ytick.minor.size"] = 5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def get_default_cmap():
    """Return the default colormap.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The default colormap.
    """
    return plt.get_cmap("viridis")


def get_cmap_from_name(cmap_name):
    """Return a colormap given its name.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap corresponding to the given name.
    """
    return plt.get_cmap(cmap_name)
