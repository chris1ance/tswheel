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

    def make_boxplot(
        self,
        data: pd.DataFrame,
        value_column: str,
        group_column: str | None = None,
        color: str | dict[str, str] | None = None,
        title: str = "",
        x_axis_title: str = "",
        y_axis_title: str = "",
        title_font_size: int = 24,
        axis_title_font_size: int = 20,
        tick_font_size: int = 18,
        show_outliers: bool = True,
        box_width: int = 30,
        median_color: str = "red",
        custom_order: list[str] | None = None,
        legend_title: str = "",
        legend_box_orient: Literal[LEGEND_BOX_ORIENTATIONS] = "top-left",
        legend_direction: Literal["horizontal", "vertical"] = "vertical",
        add_legend_border: bool = True,
        y_tick_min: int | float | None = None,
        y_tick_max: int | float | None = None,
        y_tick_step: int | float | None = None,
    ):
        """
        Create an Altair boxplot chart from data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing the data to plot.
        value_column : str
            Column name in the DataFrame that contains the values for the boxplot.
        group_column : str | None, default=None
            Optional column name to group the data by. If None, a single boxplot is created.
        color : str | dict[str, str] | None, default="black"
            Color of the boxplot elements. Can be:
            - A string for a single color (e.g., "black")
            - A string for a named color scheme (e.g., "category10")
            - A dictionary mapping group values to specific colors
            - None to use the default color scheme
        title : str, default=""
            Title for the chart.
        x_axis_title : str, default=""
            Title for the x-axis.
        y_axis_title : str, default=""
            Title for the y-axis.
        title_font_size : int, default=24
            Font size for the chart title.
        axis_title_font_size : int, default=20
            Font size for the axis titles.
        tick_font_size : int, default=18
            Font size for the axis tick labels.
        show_outliers : bool, default=True
            Whether to show outlier points outside the whiskers.
        box_width : int, default=30
            Width of the box in pixels.
        median_color : str, default="red"
            Color of the median line inside the boxplot.
        custom_order : list[str] | None, default=None
            Custom order for the categories if group_column is specified.
        legend_title : str, default=""
            Title for the legend when using multiple colors.
        legend_box_orient : Literal[LEGEND_BOX_ORIENTATIONS], default="top-left"
            Position of legend box in plot.
        legend_direction : Literal["horizontal", "vertical"], default="vertical"
            Direction of the legend items.
        add_legend_border : bool, default=True
            Whether to add a border around the legend.
        y_tick_min : int | float | None, default=None
            Minimum value for y-axis ticks. If None, determined automatically.
        y_tick_max : int | float | None, default=None
            Maximum value for y-axis ticks. If None, determined automatically.
        y_tick_step : int | float | None, default=None
            Step size between y-axis ticks. If None, determined automatically.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing the boxplot with specified customizations.
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()

        # Determine x encode based on whether grouping is used
        if group_column:
            if custom_order:
                x_encode = alt.X(
                    f"{group_column}:N",
                    sort=custom_order,
                    axis=alt.Axis(
                        title=x_axis_title,
                        titleFontSize=axis_title_font_size,
                        titleFontWeight="normal",
                        labelFontSize=tick_font_size,
                        grid=False,
                    ),
                )
            else:
                x_encode = alt.X(
                    f"{group_column}:N",
                    axis=alt.Axis(
                        title=x_axis_title,
                        titleFontSize=axis_title_font_size,
                        titleFontWeight="normal",
                        labelFontSize=tick_font_size,
                        grid=False,
                    ),
                )
        else:
            # If no grouping, use a constant column for x
            df["_constant"] = "Value"
            x_encode = alt.X(
                "_constant:N",
                axis=alt.Axis(
                    title=x_axis_title,
                    titleFontSize=axis_title_font_size,
                    titleFontWeight="normal",
                    labelFontSize=tick_font_size,
                    grid=False,
                ),
            )

        # Custom y-axis settings
        if (
            y_tick_min is not None
            and y_tick_max is not None
            and y_tick_step is not None
        ):
            yticks = list(range(y_tick_min, y_tick_max + y_tick_step, y_tick_step))
            y_encode = alt.Y(
                f"{value_column}:Q",
                scale=alt.Scale(domain=[yticks[0], yticks[-1]]),
                axis=alt.Axis(
                    values=yticks,
                    title=y_axis_title,
                    titleFontSize=axis_title_font_size,
                    titleFontWeight="normal",
                    labelFontSize=tick_font_size,
                    gridDash=[2, 2],
                    gridColor="darkgray",
                ),
            )
        else:
            y_encode = alt.Y(
                f"{value_column}:Q",
                axis=alt.Axis(
                    title=y_axis_title,
                    titleFontSize=axis_title_font_size,
                    titleFontWeight="normal",
                    labelFontSize=tick_font_size,
                    gridDash=[2, 2],
                    gridColor="darkgray",
                ),
            )

        # Handle different color configurations
        if group_column and color:
            if isinstance(color, dict):
                # Use custom color mapping for groups
                group_values = list(color.keys())
                color_values = list(color.values())
                color_scale = alt.Scale(domain=group_values, range=color_values)

                # Create legend configuration
                legend = alt.Legend(
                    title=legend_title,
                    titleFontSize=axis_title_font_size,
                    labelFontSize=tick_font_size,
                    orient=legend_box_orient,
                    direction=legend_direction,
                    strokeColor="black" if add_legend_border else None,
                    fillColor="white",
                )

                color_encode = alt.Color(
                    f"{group_column}:N", scale=color_scale, legend=legend
                )

                # Create the boxplot with color encoding
                boxplot = (
                    alt.Chart(df)
                    .mark_boxplot(
                        size=box_width,
                        outliers=show_outliers,
                        median={"color": median_color},
                        ticks=True,
                    )
                    .encode(
                        x=x_encode,
                        y=y_encode,
                        color=color_encode,
                    )
                )
            else:
                # Use a color scheme
                boxplot = (
                    alt.Chart(df)
                    .mark_boxplot(
                        size=box_width,
                        outliers=show_outliers,
                        median={"color": median_color},
                        ticks=True,
                    )
                    .encode(
                        x=x_encode,
                        y=y_encode,
                        color=alt.Color(
                            f"{group_column}:N",
                            scale=alt.Scale(scheme=color),
                            legend=alt.Legend(
                                title=legend_title,
                                titleFontSize=axis_title_font_size,
                                labelFontSize=tick_font_size,
                                orient=legend_box_orient,
                                direction=legend_direction,
                                strokeColor="black" if add_legend_border else None,
                                fillColor="white",
                            ),
                        ),
                    )
                )
        else:
            # Single color boxplot
            boxplot = (
                alt.Chart(df)
                .mark_boxplot(
                    size=box_width,
                    outliers=show_outliers,
                    median={"color": median_color},
                    ticks=True,
                    color=color if isinstance(color, str) else "black",
                )
                .encode(
                    x=x_encode,
                    y=y_encode,
                )
            )

        chart = boxplot

        # Set title and dimensions
        alt_title = alt.TitleParams(
            text=title, anchor="middle", fontSize=title_font_size
        )

        chart = chart.properties(width=self.width, height=self.height, title=alt_title)

        return chart


if __name__ == "__main__":
    pass
