"""Utilities for Altair histogram plots."""

import pandas as pd
import altair as alt
from typing import Literal

pd.set_option("mode.copy_on_write", True)


class HistogramPlotter:
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

    def __init__(
        self,
        width: int = 800,
        height: int = 400,
    ):
        self.width = width
        self.height = height

    def set_width(self, width: int):
        """Set the width of the plot."""
        self.width = width

    def set_height(self, height: int):
        """Set the height of the plot."""
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
        bins : int, default=10
            Number of bins to use in the histogram.
        bin_step : float, optional
            Width of each bin. If provided, overrides the bins parameter.
        bin_min : float, optional
            Minimum value for binning. If None, uses the minimum value in the data.
        bin_max : float, optional
            Maximum value for binning. If None, uses the maximum value in the data.
        color : str, default='steelblue'
            Color of the histogram bars when not using groupby.
        opacity : float, default=0.8
            Opacity of the histogram bars.
        groupby_column : str, optional
            Column name to group by, creating separate histograms for each group.
        group_colors : dict[str, str] | str, optional
            Colors for each group. Can be a dictionary mapping group values to colors,
            or a string specifying an Altair color scheme.
        title : str, default=""
            Title for the chart.
        x_axis_title : str, default=""
            Title for the x-axis.
        y_axis_title : str, default="Count"
            Title for the y-axis.
        legend_title : str, default=""
            Title for the legend.
        add_legend_border : bool, default=True
            Whether to add a border around the legend.
        legend_box_orient : str, default="top-right"
            Position of the legend within the chart.
        legend_direction : str, default="vertical"
            Direction of the legend items.
        title_font_size : int, default=24
            Font size for the chart title.
        axis_title_font_size : int, default=20
            Font size for the axis titles.
        tick_font_size : int, default=18
            Font size for the axis tick labels.
        x_ticks_angle : int, default=0
            Angle for the x-axis tick labels.
        show_mean_line : bool, default=False
            Whether to show a vertical line at the mean value.
        show_median_line : bool, default=False
            Whether to show a vertical line at the median value.
        mean_line_color : str, default="red"
            Color of the mean line.
        median_line_color : str, default="green"
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

            mean_rule = (
                alt.Chart(pd.DataFrame({"mean": [mean_value]}))
                .mark_rule(color=mean_line_color, strokeDash=[4, 4], size=4)
                .encode(
                    x="mean:Q",
                )
            )

            # Add text label for mean
            mean_text = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "mean": [mean_value],
                            "y": [0],
                            "label": [f"Mean: {mean_value:.2f}"],
                        }
                    )
                )
                .mark_text(
                    align="right",
                    dx=-10,  # Offset text to the right of the line
                    dy=-self.height * 0.95,  # Offset text from the bottom
                    fontSize=tick_font_size,
                    color=mean_line_color,
                    fontWeight="bold",
                )
                .encode(x="mean:Q", y="y:Q", text="label:N")
            )
            chart = chart + mean_rule + mean_text

        # Add median line if requested
        if show_median_line:
            median_value = df[value_column].median()

            median_rule = (
                alt.Chart(pd.DataFrame({"median": [median_value]}))
                .mark_rule(color=median_line_color, strokeDash=[4, 4], size=4)
                .encode(
                    x="median:Q",
                )
            )

            # Add text label for median
            median_text = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "median": [median_value],
                            "y": [0],
                            "label": [f"Median: {median_value:.2f}"],
                        }
                    )
                )
                .mark_text(
                    align="left",
                    dx=10,  # Offset text to the right of the line
                    dy=-self.height * 0.95,  # Offset text from the bottom
                    fontSize=tick_font_size,
                    color=median_line_color,
                    fontWeight="bold",
                )
                .encode(x="median:Q", y="y:Q", text="label:N")
            )

            chart = chart + median_rule + median_text

        # Set title and dimensions
        alt_title = alt.TitleParams(
            text=title, anchor="middle", fontSize=title_font_size
        )

        chart = chart.properties(width=self.width, height=self.height, title=alt_title)

        return chart


if __name__ == "__main__":
    pass
