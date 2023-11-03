"""Plotting-related small utilities."""

__all__ = ["pause",
           "link_3d_subplot_cameras"]

import matplotlib as mpl
import matplotlib.pyplot as plt


# This is the same as `extrafeathers.plotmagic.pause`, but we don't want to depend on `extrafeathers`,
# because the project scopes are different, and it also pulls in FEniCS, which is rather large.
def pause(interval: float) -> None:
    """Redraw the current Matplotlib figure **without stealing focus**.

    **IMPORTANT**:

    Works after `plt.show()` has been called at least once.

    **Background**:

    Matplotlib (3.3.3) has a habit of popping the figure window to top when it
    is updated using show() or pause(), which effectively prevents using the
    machine for anything else while a simulation is in progress.

    On some systems, it helps to force Matplotlib to use the "Qt5Agg" backend:
        https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus

    but on some others (Linux Mint 20.1) also that backend steals focus.

    One option is to use a `FuncAnimation`, but it has a different API from
    just plotting regularly, and it is not applicable in all cases.

    So, we provide this a custom non-focus-stealing pause function hack,
    based on the StackOverflow answer by user @ImportanceOfBeingErnest:
        https://stackoverflow.com/a/45734500
    """
    backend = plt.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)


def link_3d_subplot_cameras(fig, axes):
    """Link view angles and zooms of several 3d subplots in the same Matplotlib figure.

    `fig`: a Matplotlib figure object
    `axes`: a list of Matplotlib `Axes` objects in figure `fig`

    Return value is a handle to a Matplotlib `motion_notify_event`;
    you can disconnect this event later if you want to unlink the cameras.

    Example::

        fig = plt.figure(1)
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax2 = fig.add_subplot(2, 3, 1, projection="3d")
        link_3d_subplot_cameras(fig, [ax1, ax2])

    Adapted from recipe at:
        https://github.com/matplotlib/matplotlib/issues/11181
    """
    def on_move(event):
        sender = [ax for ax in axes if event.inaxes == ax]
        if not sender:
            return
        assert len(sender) == 1
        sender = sender[0]
        others = [ax for ax in axes if ax is not sender]
        if sender.button_pressed in sender._rotate_btn:
            for ax in others:
                ax.view_init(elev=sender.elev, azim=sender.azim)
        elif sender.button_pressed in sender._zoom_btn:
            for ax in others:
                ax.set_xlim3d(sender.get_xlim3d())
                ax.set_ylim3d(sender.get_ylim3d())
                ax.set_zlim3d(sender.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()
    return fig.canvas.mpl_connect("motion_notify_event", on_move)
