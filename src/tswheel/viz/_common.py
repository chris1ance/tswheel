"""Common constants used throughout the tswheel package."""

# Shared constants for chart/plot configurations
LEGEND_BOX_ORIENTATIONS = [
    "none",
    "left",
    "right",
    "top",
    "bottom",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
]


class BasePlotter:
    """
    Base class for all plotters in the tswheel package.

    This class provides common functionality for setting plot dimensions and font sizes
    that are used across different plotter implementations.
    """

    LEGEND_BOX_ORIENTATIONS = LEGEND_BOX_ORIENTATIONS

    def __init__(
        self,
        width: int = 800,
        height: int = 400,
        title_font_size: int = 24,
        axis_title_font_size: int = 20,
        tick_font_size: int = 18,
    ):
        """
        Initialize a BasePlotter with specified dimensions and font sizes.

        Parameters:
        -----------
        width : int, default=800
            Width of the plot in pixels.
        height : int, default=400
            Height of the plot in pixels.
        title_font_size : int, default=24
            Font size for the chart title.
        axis_title_font_size : int, default=20
            Font size for the axis titles.
        tick_font_size : int, default=18
            Font size for the axis tick labels.
        """
        self.width = width
        self.height = height
        self.title_font_size = title_font_size
        self.axis_title_font_size = axis_title_font_size
        self.tick_font_size = tick_font_size

    def set_width(self, width: int):
        """
        Set the width of the plot.

        Parameters:
        -----------
        width : int
            New width value in pixels.
        """
        self.width = width

    def set_height(self, height: int):
        """
        Set the height of the plot.

        Parameters:
        -----------
        height : int
            New height value in pixels.
        """
        self.height = height

    def set_title_font_size(self, title_font_size: int):
        """
        Set the font size for chart titles.

        Parameters:
        -----------
        title_font_size : int
            New font size for chart titles.
        """
        self.title_font_size = title_font_size

    def set_axis_title_font_size(self, axis_title_font_size: int):
        """
        Set the font size for axis titles.

        Parameters:
        -----------
        axis_title_font_size : int
            New font size for axis titles.
        """
        self.axis_title_font_size = axis_title_font_size

    def set_tick_font_size(self, tick_font_size: int):
        """
        Set the font size for axis tick labels.

        Parameters:
        -----------
        tick_font_size : int
            New font size for axis tick labels.
        """
        self.tick_font_size = tick_font_size

    def make_zero_hline_plot(self, yticks: list[int | float]):
        """
        Create a horizontal line at y=0 for charts that include both positive and negative values.

        Parameters:
        -----------
        yticks : list[int | float]
            List of y-axis tick values that define the plot's y-axis scale domain.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing a black horizontal line at y=0.
            The line spans the full width of the plot and has a thickness of 3 pixels.

        Notes:
        ------
        This function is typically used as an overlay on other charts to clearly demarcate
        the boundary between positive and negative values.
        """
        import pandas as pd
        import altair as alt

        plot = (
            alt.Chart(pd.DataFrame({"Value": [0]}))
            .mark_rule(color="black", size=3)
            .encode(
                y=alt.Y(
                    "Value:Q",
                    scale=alt.Scale(domain=[yticks[0], yticks[-1]]),
                    axis=alt.Axis(values=yticks),
                )
            )
        )

        return plot
