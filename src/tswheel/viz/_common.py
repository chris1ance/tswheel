"""Common constants used throughout the tswheel package."""

import pandas as pd
import altair as alt
import numpy as np

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

    def make_zero_hline_chart(
        self,
        yticks: list[int | float] | None = None,
        y_tick_min: float | None = None,
        y_tick_max: float | None = None,
        y_tick_step: float | None = None,
        line_color: str = "black",
        line_size: int = 3,
    ):
        """
        Create a horizontal line at y=0 for charts that include both positive and negative values.

        This method provides flexibility in specifying the y-axis domain through either:
        1. A list of y-axis ticks
        2. Minimum and maximum y values
        3. All three: yticks, min, and max values

        Parameters:
        -----------
        yticks : list[int | float] | None, default=None
            List of y-axis tick values that define the plot's y-axis scale domain.
        y_tick_min : float | None, default=None
            Minimum value for the y-axis domain.
        y_tick_max : float | None, default=None
            Maximum value for the y-axis domain.
        y_tick_step : float | None, default=None
            Step size between ticks when generating ticks from min/max values.
        line_color : str, default="black"
            Color of the horizontal line.
        line_size : int, default=3
            Thickness of the horizontal line in pixels.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing a horizontal line at y=0.

        Raises:
        -------
        ValueError
            If neither yticks nor both y_tick_min and y_tick_max are provided.

        Examples:
        ---------
        >>> # Using a list of ticks
        >>> plotter = BasePlotter()
        >>> zero_line = plotter.make_zero_hline_chart(yticks=[-10, -5, 0, 5, 10])
        >>>
        >>> # Using min/max values
        >>> zero_line = plotter.make_zero_hline_chart(y_tick_min=-10, y_tick_max=10)
        >>>
        >>> # Using min/max with step to generate ticks
        >>> zero_line = plotter.make_zero_hline_chart(
        ...     y_tick_min=-10, y_tick_max=10, y_tick_step=2
        ... )
        """
        # Validate that we have the necessary parameters for at least one approach
        if yticks is None and (y_tick_min is None or y_tick_max is None):
            raise ValueError(
                "Either yticks or both y_tick_min and y_tick_max must be provided"
            )

        # Case 1: Using yticks list
        if (isinstance(yticks, list)) and (len(yticks) > 0):
            domain_min = yticks[0]
            domain_max = yticks[-1]
            axis_values = yticks
        # Case 2: Using min/max with step to generate ticks
        elif y_tick_step is not None:
            # Generate yticks from min/max/step
            yticks_list = list(
                np.arange(y_tick_min, y_tick_max + y_tick_step, y_tick_step)
            )
            domain_min = yticks_list[0]
            domain_max = yticks_list[-1]
            axis_values = yticks_list

        # Case 3: Using only min/max values
        else:
            domain_min = y_tick_min
            domain_max = y_tick_max
            axis_values = None  # No axis_values specified in this case

        # Create the chart with the calculated domain and axis values
        y_encoding = alt.Y(
            "Value:Q",
            scale=alt.Scale(domain=[domain_min, domain_max]),
        )

        # Add axis values if they were determined
        if axis_values is not None:
            y_encoding = alt.Y(
                "Value:Q",
                scale=alt.Scale(domain=[domain_min, domain_max]),
                axis=alt.Axis(values=axis_values),
            )

        return (
            alt.Chart(pd.DataFrame({"Value": [0]}))
            .mark_rule(color=line_color, size=line_size)
            .encode(y=y_encoding)
        )
