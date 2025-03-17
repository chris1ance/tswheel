"""Utilities for Altair heatmap plots."""

import pandas as pd
import altair as alt

from ._common import BasePlotter

pd.set_option("mode.copy_on_write", True)


class HeatmapPlotter(BasePlotter):
    """
    A class for creating customized heatmap plots using Altair.

    This class provides methods to create and customize heatmap visualizations.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 400,
        title_font_size: int = 24,
        axis_title_font_size: int = 20,
        tick_font_size: int = 18,
    ):
        """
        Initialize a HeatmapPlotter with specified dimensions and font sizes.

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
        super().__init__(
            width=width,
            height=height,
            title_font_size=title_font_size,
            axis_title_font_size=axis_title_font_size,
            tick_font_size=tick_font_size,
        )

    def make_heatmap(
        self,
        data: pd.DataFrame,
        text_fontsize: int = 14,
        title: str = "",
        x_axis_title: str = "",
        y_axis_title: str = "",
        x_ticks_angle: int = 0,
    ):
        """
        Create an Altair heatmap chart from tabular data.

        This method transforms the input DataFrame into a format suitable for heatmap visualization,
        where rows represent the x-axis, columns represent the y-axis, and cell values determine
        the color intensity.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing the data to visualize as a heatmap. The row index values will
            be used for the x-axis, and column names for the y-axis. Cell values determine
            the color intensity in the heatmap.
        text_fontsize : int, default=14
            Font size for the text displayed in each cell of the heatmap.
        title : str, default=""
            Title for the chart.
        x_axis_title : str, default=""
            Title for the x-axis.
        y_axis_title : str, default=""
            Title for the y-axis.
        x_ticks_angle : int, default=0
            Angle to rotate x-axis labels in degrees. Useful when x-axis labels are long.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing the heatmap with specified customizations.
            The chart includes both the colored rectangles and the text values.

        Examples:
        ---------
        >>> import pandas as pd
        >>> from tswheel.viz import HeatmapPlotter
        >>>
        >>> # Create sample correlation data
        >>> data = pd.DataFrame({
        ...     'A': [1.0, 0.7, 0.3],
        ...     'B': [0.7, 1.0, 0.2],
        ...     'C': [0.3, 0.2, 1.0]
        ... }, index=['A', 'B', 'C'])
        >>>
        >>> # Create the heatmap
        >>> plotter = HeatmapPlotter(width=600, height=600)
        >>> chart = plotter.make_heatmap(
        ...     data=data,
        ...     title="Correlation Matrix",
        ...     x_axis_title="Variables",
        ...     y_axis_title="Variables"
        ... )
        """
        df = data.reset_index(names="Index").melt(
            id_vars="Index", var_name="Columns", value_name="Value"
        )

        alt_X = alt.X(
            "Index:O",
            sort=None,
            title=x_axis_title,
            axis=alt.Axis(labelAngle=x_ticks_angle),
        )

        alt_Y = alt.Y("Columns:O", sort=None, title=y_axis_title)

        heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt_X,
                y=alt_Y,
                color=alt.Color(
                    "Value:Q",
                    legend=None,
                    scale=alt.Scale(
                        scheme="lightmulti",
                    ),
                ),
            )
        )

        # Configure text
        heatmap_text = (
            alt.Chart(df)
            .mark_text(baseline="middle", fontWeight="bold", fontSize=text_fontsize)
            .encode(
                x=alt_X,
                y=alt_Y,
                text="Value:Q",
            )
        )

        alt_TitleParams = alt.TitleParams(text=title, fontSize=self.title_font_size)
        chart = (heatmap + heatmap_text).properties(
            width=self.width, height=self.height, title=alt_TitleParams
        )

        return chart
