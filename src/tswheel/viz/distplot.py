"""Utilities for Altair distribution plots."""

import pandas as pd
import numpy as np
import altair as alt
from altair.utils.schemapi import Undefined
from typing import Literal

from ._common import BasePlotter

pd.set_option("mode.copy_on_write", True)


class DistributionPlotter(BasePlotter):
    """
    A class for creating customized distribution plots using Altair.

    This class provides methods to create and customize histogram visualizations,
    including options for adding statistical indicators like mean and median lines.
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
        Initialize a DistributionPlotter with specified dimensions and font sizes.

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

        Examples:
        ---------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tswheel.viz import DistributionPlotter
        >>>
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({'values': np.random.normal(0, 1, 1000)})
        >>>
        >>> # Create a histogram with mean and median lines
        >>> plotter = DistributionPlotter(width=700, height=400)
        >>> chart = plotter.make_histogram(
        ...     data=data,
        ...     value_column='values',
        ...     bin_step=0.2,
        ...     title="Normal Distribution",
        ...     x_axis_title="Value",
        ...     show_mean_line=True,
        ...     show_median_line=True
        ... )
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
                titleFontSize=self.axis_title_font_size,
                labelFontSize=self.tick_font_size,
                grid=False,
                labelAlign="center",  # Centers labels under their tick marks
                format=f".{x_tick_decimal_places}f",
            ),
        )

        alt_y = alt.Y(
            "count()",
            axis=alt.Axis(
                title=y_axis_title,
                titleFontSize=self.axis_title_font_size,
                labelFontSize=self.tick_font_size,
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
                text_font_size=self.tick_font_size,
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
                text_font_size=self.tick_font_size,
                line_color=median_line_color,
            )
            chart = chart + median_chart

        # Set title and dimensions
        alt_title = alt.TitleParams(
            text=title, anchor="middle", fontSize=self.title_font_size
        )

        chart = chart.properties(width=self.width, height=self.height, title=alt_title)

        return chart

    def make_boxplot(
        self,
        data: pd.DataFrame,
        value_column: str,
        group_column: str,
        series_colors: str | dict[str, str] | None = None,
        title: str = "",
        x_axis_title: str = "",
        y_axis_title: str = "",
        show_outliers: bool = False,
        box_width: int = 30,
        median_color: str = "black",
        median_line_width: int = 2,
        extent: float = 1.5,
        custom_order: list[str] | None = None,
        y_tick_min: int | float | None = None,
        y_tick_max: int | float | None = None,
        y_tick_step: int | float | None = None,
        x_ticks_angle: int = 0,
        facet_column: str | None = None,
        facet_order: list[str] | None = None,
        columns: int | None = None,
    ):
        """
        Create an Altair boxplot chart from data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing the data to plot.
        value_column : str
            Column name in the DataFrame that contains the values for the boxplot.
        group_column : str
            Column name to group the data by. For a single boxplot, specify a constant column.
        series_colors : dict[str, str] | str | None, optional
            Dictionary mapping series names to their desired colors, a named color scheme,
            or None to use the default 'category10' color scheme. Default: None.
        title : str, default=""
            Title for the chart.
        x_axis_title : str, default=""
            Title for the x-axis.
        y_axis_title : str, default=""
            Title for the y-axis.
        show_outliers : bool, default=False
            Whether to show outlier points outside the whiskers.
        box_width : int, default=30
            Width of the box in pixels.
        median_color : str, default="black"
            Color of the median line inside the boxplot.
        median_line_width : int, default=2
            Width of the median line inside the boxplot.
        extent : float, default=1.5
            How far the whiskers extend from the box as a multiple of the IQR.
        custom_order : list[str] | None, default=None
            Custom order for the categories if group_column is specified.
        y_tick_min : int | float | None, default=None
            Minimum value for y-axis ticks. If None, determined automatically.
        y_tick_max : int | float | None, default=None
            Maximum value for y-axis ticks. If None, determined automatically.
        y_tick_step : int | float | None, default=None
            Step size between y-axis ticks. If None, determined automatically.
        x_ticks_angle : int, default=0
            Angle to rotate x-axis labels.
        facet_column : str | None, default=None
            Column name to facet the data by, creating multiple charts.
            If None, no faceting is applied.
        facet_order : list[str] | None, default=None
            Custom order for the facet categories if facet_column is specified.
        columns : int | None, default=None
            Maximum number of columns in the faceted layout. If None, Altair
            will determine the number of columns automatically.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing the boxplot with specified customizations.

        Examples:
        ---------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tswheel.viz import DistributionPlotter
        >>>
        >>> # Create sample data with multiple groups and facets
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({
        ...     'group': np.repeat(['A', 'B', 'C'], 100),
        ...     'facet': np.repeat(['X', 'Y', 'Z', 'W'], 75),
        ...     'value': np.concatenate([
        ...         np.random.normal(0, 1, 100),  # Group A
        ...         np.random.normal(2, 1.5, 100),  # Group B
        ...         np.random.normal(-1, 0.5, 100)  # Group C
        ...     ])
        ... })
        >>>
        >>> # Create faceted boxplots with 2 columns
        >>> plotter = DistributionPlotter(width=600, height=400)
        >>> chart = plotter.make_boxplot(
        ...     data=data,
        ...     value_column='value',
        ...     group_column='group',
        ...     facet_column='facet',
        ...     columns=2,
        ...     title="Distribution by Group and Facet",
        ...     x_axis_title="Group",
        ...     y_axis_title="Value"
        ... )
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()

        # Custom x-axis settings
        x_encode = alt.X(
            f"{group_column}:N",
            sort=custom_order if custom_order else Undefined,
            axis=alt.Axis(
                titleFontSize=self.axis_title_font_size,
                titleFontWeight="normal",
                labelFontSize=self.tick_font_size,
                labelAngle=x_ticks_angle,
                grid=False,
            ),
            title=x_axis_title,
        )

        # Custom y-axis settings
        if (
            y_tick_min is not None
            and y_tick_max is not None
            and y_tick_step is not None
        ):
            yticks = list(np.arange(y_tick_min, y_tick_max + y_tick_step, y_tick_step))
            scale = alt.Scale(domain=[yticks[0], yticks[-1]])
        else:
            yticks = Undefined
            if y_tick_min is not None and y_tick_max is not None:
                scale = alt.Scale(domain=[y_tick_min, y_tick_max])
            else:
                scale = Undefined

        y_encode = alt.Y(
            f"{value_column}:Q",
            scale=scale,
            axis=alt.Axis(
                values=yticks,
                titleFontSize=self.axis_title_font_size,
                titleFontWeight="normal",
                labelFontSize=self.tick_font_size,
                gridDash=[2, 2],
                gridColor="darkgray",
            ),
            title=y_axis_title,
        )

        # Handle different color configurations
        if isinstance(series_colors, dict):
            # Use custom color mapping for groups
            group_values = list(series_colors.keys())
            color_values = list(series_colors.values())
            color_scale = alt.Scale(domain=group_values, range=color_values)

            color_encode = alt.Color(
                f"{group_column}:N", scale=color_scale, legend=None
            )

        else:
            color_encode = alt.Color(
                f"{group_column}:N",
                scale=alt.Scale(scheme=series_colors if series_colors else "plasma"),
                legend=None,
            )

        # Create the boxplot
        chart = (
            alt.Chart(df)
            .mark_boxplot(
                size=box_width,
                outliers=show_outliers,
                median={"color": median_color, "strokeWidth": median_line_width},
                ticks=False,
                extent=extent,
            )
            .encode(
                x=x_encode,
                y=y_encode,
                color=color_encode,
            )
        )

        # Set dimensions
        chart = chart.properties(width=self.width, height=self.height)

        # Add zero line if the y-axis range includes both positive and negative values
        # Case 1: Explicit yticks list that spans zero
        has_zero_line = False
        if (
            isinstance(yticks, list)
            and any(y < 0 for y in yticks)
            and any(y > 0 for y in yticks)
        ):
            zero_line = self.make_zero_hline_chart(yticks=yticks)
            chart += zero_line
            has_zero_line = True
        # Case 2: Min/max values that span zero
        elif (
            y_tick_min is not None
            and y_tick_max is not None
            and y_tick_min < 0 < y_tick_max
        ):
            zero_line = self.make_zero_hline_chart(
                y_tick_min=y_tick_min, y_tick_max=y_tick_max, y_tick_step=y_tick_step
            )
            chart += zero_line
            has_zero_line = True

        # Apply faceting if a facet column is provided
        if facet_column:
            # Create a sort parameter for facet order if provided
            sort_param = facet_order if facet_order else Undefined

            # Prepare common facet parameters
            facet_params = {
                "facet": alt.Facet(
                    f"{facet_column}:N",
                    sort=sort_param,
                    title="",
                    header=alt.Header(
                        titleFontSize=self.axis_title_font_size,
                        labelFontSize=self.tick_font_size,
                    ),
                )
            }

            # Add optional parameters only when they are specified
            if columns is not None:
                facet_params["columns"] = columns

            # For layered charts (when zero line is added), we need to pass the data explicitly
            # to avoid Altair error: "Facet charts require data to be specified at the top level"
            if has_zero_line:
                # Include data parameter for layered charts
                facet_params["data"] = df

            # Apply faceting with the prepared parameters
            chart = chart.facet(**facet_params)

        # Set title
        alt_title = alt.TitleParams(
            text=title, anchor="middle", fontSize=self.title_font_size
        )

        chart = chart.properties(title=alt_title)

        return chart


if __name__ == "__main__":
    pass
