"""Utilities for Altair histogram plots."""

import pandas as pd
import altair as alt
from typing import Literal

from .constants import LEGEND_BOX_ORIENTATIONS

pd.set_option("mode.copy_on_write", True)


class HistogramPlotter:
    """
    A class for creating customized histogram plots using Altair.

    This class provides methods to create and customize histogram visualizations,
    including options for adding statistical indicators like mean and median lines.
    """

    LEGEND_BOX_ORIENTATIONS = LEGEND_BOX_ORIENTATIONS

    def __init__(
        self,
        width: int = 800,
        height: int = 400,
    ):
        """
        Initialize a HistogramPlotter with specified dimensions.

        Parameters:
        -----------
        width : int, default=800
            Width of the plot in pixels.
        height : int, default=400
            Height of the plot in pixels.
        """
        self.width = width
        self.height = height

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

    def make_vline_plot(
        self,
        name: str,
        value: int | float,
        align: Literal["right", "left"],
        text_font_size: int,
        line_color: str,
        height_multiplier: float = 0.95,
    ):
        """
        Create a vertical line with label for a statistical value.

        Parameters:
        -----------
        name : str
            Name of the statistical value (e.g., "Mean", "Median").
        value : int | float
            The value at which to draw the vertical line.
        align : Literal["right", "left"]
            Text alignment relative to the line.
        text_font_size : int
            Font size for the text label.
        line_color : str
            Color of the vertical line.
        height_multiplier : float, default=0.95
            Multiplier that determines the vertical position of the label.

        Returns:
        --------
        alt.Chart
            Altair chart object containing the vertical line and label.
        """
        rule = (
            alt.Chart(pd.DataFrame({name: [value]}))
            .mark_rule(color=line_color, strokeDash=[4, 4], size=4)
            .encode(
                x=f"{name}:Q",
            )
        )

        text = (
            alt.Chart(
                pd.DataFrame(
                    {
                        name: [value],
                        "y": [0],
                        "label": [f"{name}: {value:.2f}"],
                    }
                )
            )
            .mark_text(
                align=align,
                dx=10 if align == "left" else -10,  # Offset text from the line
                dy=-self.height * height_multiplier,  # Offset text from the bottom
                fontSize=text_font_size,
                color=line_color,
                fontWeight="bold",
            )
            .encode(x=f"{name}:Q", y="y:Q", text="label:N")
        )

        chart = rule + text

        return chart

    def make_histogram(
        self,
        data: pd.DataFrame,
        value_column: str,
        bin_step: float,
        color: str = "black",
        opacity: float = 1.0,
        title: str = "",
        x_axis_title: str = "",
        y_axis_title: str = "Count",
        title_font_size: int = 24,
        axis_title_font_size: int = 20,
        tick_font_size: int = 18,
        x_tick_decimal_places: int = 1,
        show_mean_line: bool = False,
        show_median_line: bool = False,
        mean_line_color: str = "orange",
        median_line_color: str = "red",
    ):
        """
        Create an Altair histogram chart from data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing the data to plot.
        value_column : str
            Column name in the DataFrame that contains the values to create a histogram from.
        bin_step : float
            Width of each bin.
        color : str, default="black"
            Color of the histogram bars.
        opacity : float, default=1.0
            Opacity of the histogram bars.
        title : str, default=""
            Title for the chart.
        x_axis_title : str, default=""
            Title for the x-axis.
        y_axis_title : str, default="Count"
            Title for the y-axis.
        title_font_size : int, default=24
            Font size for the chart title.
        axis_title_font_size : int, default=20
            Font size for the axis titles.
        tick_font_size : int, default=18
            Font size for the axis tick labels.
        x_tick_decimal_places : int, default=1
            Number of decimal places to show in x-axis tick labels.
        show_mean_line : bool, default=False
            Whether to show a vertical line at the mean value.
        show_median_line : bool, default=False
            Whether to show a vertical line at the median value.
        mean_line_color : str, default="orange"
            Color of the mean line.
        median_line_color : str, default="red"
            Color of the median line.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing the histogram with specified customizations.
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()

        # Calculate bin settings
        # For bin params see: https://altair-viz.github.io/user_guide/generated/core/altair.BinParams.html#altair.BinParams
        bin_settings = {
            "binned": False,
            "step": bin_step,
        }

        # Custom x and y axis settings
        alt_x = alt.X(
            f"{value_column}:Q",
            bin=bin_settings,
            axis=alt.Axis(
                title=x_axis_title,
                titleFontSize=axis_title_font_size,
                labelFontSize=tick_font_size,
                grid=False,
                labelAlign="center",  # Centers labels under their tick marks
                format=f".{x_tick_decimal_places}f",
            ),
        )

        alt_y = alt.Y(
            "count()",
            axis=alt.Axis(
                title=y_axis_title,
                titleFontSize=axis_title_font_size,
                labelFontSize=tick_font_size,
                gridDash=[2, 2],
                gridColor="darkgray",
            ),
        )

        chart = (
            alt.Chart(df)
            .mark_bar(opacity=opacity, color=color)
            .encode(
                x=alt_x,
                y=alt_y,
            )
        )

        # Add mean line if requested
        if show_mean_line:
            mean_value = df[value_column].mean()
            mean_chart = self.make_vline_plot(
                name="Mean",
                value=mean_value,
                align="right",
                text_font_size=tick_font_size,
                line_color=mean_line_color,
            )
            chart = chart + mean_chart

        # Add median line if requested
        if show_median_line:
            median_value = df[value_column].median()
            median_chart = self.make_vline_plot(
                name="Median",
                value=median_value,
                align="left",
                text_font_size=tick_font_size,
                line_color=median_line_color,
            )
            chart = chart + median_chart

        # Set title and dimensions
        alt_title = alt.TitleParams(
            text=title, anchor="middle", fontSize=title_font_size
        )

        chart = chart.properties(width=self.width, height=self.height, title=alt_title)

        return chart


if __name__ == "__main__":
    pass
