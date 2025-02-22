"""Utilities for Altair line plots."""

from typing import Union, Literal
import pandas as pd
from fredapi import Fred
import altair as alt
from functools import lru_cache

pd.set_option("mode.copy_on_write", True)


class LinePlotter:
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
        fred_api_key: str | None = None,
    ):
        self.fred_api_key = fred_api_key
        self.recession_bars_plot = None

    @lru_cache(maxsize=None)
    def get_recessions(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        nber_based: bool = True,
    ):
        """
        Get monthly recession start and end dates from FRED.

        Parameters:
        -----------
        start_date : str, optional
            Start date for data retrieval (default: '1900-01-01') in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for data retrieval in 'YYYY-MM-DD' format
        nber_based: bool, optional, default = True
            If True: Recessions start in the month AFTER the peak in the business cycle, and end in the month of the trough.
            If False: Recessions start in the month OF the peak in the business cycle, and end in the month of the trough.

        Returns:
        --------
        pd.DataFrame
            DataFrame with 'start' and 'end' columns containing recession dates

        Example:
        -------
        from dotenv import load_dotenv
        import os
        load_dotenv()  # take environment variables
        FREDAPIKEY = os.getenv('FREDAPIKEY')

        recession_spans = get_recession_periods(FREDAPIKEY)
        """
        if not start_date:
            start_date = "1900-01-01"

        fred = Fred(api_key=self.fred_api_key)
        fred_recession_code = "USREC" if nber_based else "USRECM"

        try:
            df = (
                fred.get_series(
                    fred_recession_code,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                .rename("recession")
                .to_frame()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data from FRED with error: {e}")

        assert not df.isna().any().any()

        all_recessions = []
        one_recession = []
        in_recession = False

        for date, row in df.iterrows():
            if row["recession"] == 1:
                in_recession = True
                one_recession.append(date)
            elif row["recession"] == 0:
                if in_recession:
                    in_recession = False
                    all_recessions.append(one_recession)
                    one_recession = []

        # Handle case where data ends during recession
        if one_recession:
            all_recessions.append(one_recession)

        if not all_recessions:
            recession_spans = pd.DataFrame(columns=["start", "end"])
        else:
            recession_spans = pd.DataFrame(
                {
                    "start": [recession[0] for recession in all_recessions],
                    "end": [recession[-1] for recession in all_recessions],
                }
            )

        return recession_spans

    def make_recession_bars_plot(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        nber_based: bool = True,
    ):
        """
        Create an Altair chart layer with gray rectangular bars representing recession periods.

        Parameters:
        -----------
        start_date : str, optional
            Start date for data retrieval in 'YYYY-MM-DD' format.
            If None, defaults to '1900-01-01'.
        end_date : str, optional
            End date for data retrieval in 'YYYY-MM-DD' format.
        nber_based: bool, optional, default = True
            If True: Recessions start in the month AFTER the peak in the business cycle, and end in the month of the trough.
            If False: Recessions start in the month OF the peak in the business cycle, and end in the month of the trough.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing semi-transparent gray rectangles
            spanning the duration of each recession period.

        Notes:
        ------
        The returned chart is designed to be overlaid on time series plots
        to visually indicate recession periods. The bars have 40% opacity
        and can be layered with other charts using the + operator.
        """
        recessions = self.get_recessions(
            start_date=start_date,
            end_date=end_date,
            nber_based=nber_based,
        )

        recession_bars_plot = (
            alt.Chart(recessions)
            .mark_rect(opacity=0.4, color="gray")
            .encode(x="start:T", x2="end:T")
        )

        return recession_bars_plot

    @staticmethod
    def make_zero_hline_plot(yticks: list[Union[int, float]]):
        """
        Create a horizontal line at y=0 for charts that include both positive and negative values.

        Parameters:
        -----------
        yticks : list[Union[int,float]]
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

    @staticmethod
    def elicit_date_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize date handling in a DataFrame by ensuring a 'date' column exists
        and is in timestamp format.

        Parameters:
            df (pd.DataFrame): Input DataFrame with date information either in index
                            or in a column named 'date' or 'Date'

        Returns:
            pd.DataFrame: DataFrame with standardized date column

        Raises:
            TypeError: If input is not a pandas DataFrame
            ValueError: If no eligible date column is found
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        _df = df.copy()

        # Handle "Date" column
        if "Date" in _df:
            _df = _df.rename(columns={"Date": "date"})

        # Handle Period Index
        if isinstance(_df.index.dtype, pd.PeriodDtype):
            _df.index = _df.index.to_timestamp()
            _df = _df.reset_index(names="date")

        # Handle DateTime Index
        elif pd.api.types.is_datetime64_any_dtype(_df.index):
            if (not _df.index.name) or (
                _df.index.name and _df.index.name.lower() != "date"
            ):
                _df.index.name = "date"
            _df = _df.reset_index()

        # Handle Period column
        elif "date" in _df:
            if isinstance(_df["date"].dtype, pd.PeriodDtype):
                _df["date"] = _df["date"].dt.to_timestamp()

        # Raise exception
        else:
            raise ValueError("No eligible date column found in input dataframe.")

        return _df

    def make_line_chart(
        self,
        data: pd.DataFrame,
        series_colors: dict[str, str],
        y_tick_min: int | float,
        y_tick_max: int | float,
        y_tick_step: int | float,
        title: str = "",
        x_axis_title: str = "",
        y_axis_title: str = "",
        add_legend_border: bool = True,
        legend_box_orient: Literal[LEGEND_BOX_ORIENTATIONS] = "top-left",
        legend_direction: Literal["horizontal", "vertical"] = "vertical",
        date_format: str = "%Y",
        title_font_size: int = 24,
        axis_title_font_size: int = 20,
        tick_font_size: int = 18,
        x_ticks_angle: int = 0,
        width: int = 800,
        height: int = 400,
    ):
        """
        Create an Altair line plot from time series data with optional recession bars overlay.

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with DateTime index and columns for each time series to plot.
        series_colors : dict[str,str]
            Dictionary mapping series names to their desired colors.
        y_tick_min : int|float
            Minimum value for y-axis ticks.
        y_tick_max : int|float
            Maximum value for y-axis ticks.
        y_tick_step : int|float
            Step size between y-axis ticks.

        Returns:
        --------
        alt.Chart
            An Altair chart object containing the line plot with specified customizations.

        Notes:
        ------
        - The function automatically adds a horizontal line at y=0 if the y-axis range includes both
        positive and negative values.
        - If a FRED API key is provided, gray bars indicating recession periods will be overlaid on the plot.
        - Y-axis gridlines are dashed and dark gray, while x-axis gridlines are disabled.
        - The chart clips any marks that fall outside the specified scale domain.
        """

        _df = self.elicit_date_column(data)
        START_DATE = _df["date"].iat[0]
        END_DATE = _df["date"].iat[-1]

        # Melt the DataFrame for Altair's long format
        df_melted = _df.melt(id_vars=["date"], var_name="Series", value_name="Value")

        # Customize the x-axis
        alt_x = alt.X(
            "date:T",
            axis=alt.Axis(
                title=x_axis_title,
                titleFontSize=axis_title_font_size,
                titleFontWeight="normal",
                format=date_format,
                labelFontSize=tick_font_size,
                labelAngle=x_ticks_angle,
                grid=False,
                labelAlign="center",  # Centers labels under their tick marks
            ),
            # scale=alt.Scale(nice=True),
        )

        # Customize the y-axis
        yticks = list(range(y_tick_min, y_tick_max + y_tick_step, y_tick_step))
        alt_y = alt.Y(
            "Value:Q",
            scale=alt.Scale(domain=[yticks[0], yticks[-1]]),
            axis=alt.Axis(
                values=yticks,
                title=y_axis_title,
                titleFontSize=axis_title_font_size,
                titleFontWeight="normal",
                labelFontSize=tick_font_size,
                titleAnchor="start",  # Puts y-axis title in bottom left corner of plot
                titleAngle=0,  # Makes y-axis title horizontal
                titleY=-10,  # Moves y-axis title up to the upper left corner of plot
                gridDash=[2, 2],
                gridColor="darkgray",
            ),
        )

        # Customize series legend
        alt_legend = alt.Legend(
            title=None,
            labelFontSize=axis_title_font_size,
            offset=1,  # Offset in pixels by which to displace the legend from the data rectangle and axes.
            symbolSize=300,  # Length of the variable’s stroke in the legend
            symbolStrokeWidth=10,  # Width of the variable’s stroke in the legend
            orient=legend_box_orient,  # Position of legend box in plot
            labelLimit=0,  # Ensures labels are not truncated
            strokeColor="black"
            if add_legend_border
            else None,  # Color of border around the legend
            fillColor="white",  # Background color of the legend box
            symbolType="stroke",  # Shape of the legend symbol
            direction=legend_direction,
        )

        # Series colors and series legend
        series_names = list(series_colors.keys())
        series_colors = list(series_colors.values())
        alt_scale = alt.Scale(domain=series_names, range=series_colors)
        alt_color = alt.Color("Series:N", scale=alt_scale, legend=alt_legend)

        # Make chart
        # clip=True: Clip marks (e.g. lines or points) falling outside specified scale domain
        chart = (
            alt.Chart(df_melted)
            .mark_line(size=4, clip=True)
            .encode(x=alt_x, y=alt_y, color=alt_color)
        )

        if self.fred_api_key:
            recession_bars_plot = self.make_recession_bars_plot(
                start_date=START_DATE, end_date=END_DATE
            )

            chart += recession_bars_plot

        if any([y < 0 for y in yticks]):
            black_hline_plot = self.make_zero_hline_plot(yticks)
            chart += black_hline_plot

        # Customize plot title
        alt_title = alt.TitleParams(text=title, fontSize=title_font_size)
        chart = chart.properties(width=width, height=height, title=alt_title)

        return chart

    def make_percentile_area_chart(
        self,
        data: pd.DataFrame,
        area_color: str,
        y_tick_min: int | float,
        y_tick_max: int | float,
        y_tick_step: int | float,
        title: str = "",
        percentile_type: Literal["25_75", "10_90", "min_max"] = "25_75",
        base_chart: alt.Chart | None = None,
        legend_box_orient: Literal[LEGEND_BOX_ORIENTATIONS] | None = None,
    ):
        """
        Create an Altair chart displaying the median with percentile ranges as an area.

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with DateTime index and a single column for the time series.
        median_color : str
            Color for the median line.
        area_color : str
            Color for the percentile area.
        y_tick_min : int|float
            Minimum value for y-axis ticks.
        y_tick_max : int|float
            Maximum value for y-axis ticks.
        y_tick_step : int|float
            Step size between y-axis ticks.
        percentile_type : Literal["25_75", "10_90", "min_max"], optional
            Type of percentile range to display.
            "25_75": 25th to 75th percentile.
            "10_90": 10th to 90th percentile.
            "min_max": Minimum to Maximum values.
            Default: "25_75".

        Returns:
        --------
        alt.Chart
            An Altair chart object containing the percentile area plot with specified customizations.

        Notes:
        ------
        - The function automatically adds a horizontal line at y=0 if the y-axis range includes both
          positive and negative values.
        - If a FRED API key is provided, gray bars indicating recession periods will be overlaid on the plot.
        - Y-axis gridlines are dashed and dark gray, while x-axis gridlines are disabled.
        - The chart clips any marks that fall outside the specified scale domain.
        """

        _df = self.elicit_date_column(data)

        # Calculate percentiles across columns
        if percentile_type == "25_75":
            lower_percentile = _df.drop(columns=["date"]).quantile(0.25, axis=1)
            upper_percentile = _df.drop(columns=["date"]).quantile(0.75, axis=1)
        elif percentile_type == "10_90":
            lower_percentile = _df.drop(columns=["date"]).quantile(0.10, axis=1)
            upper_percentile = _df.drop(columns=["date"]).quantile(0.90, axis=1)
        elif percentile_type == "min_max":
            lower_percentile = _df.drop(columns=["date"]).min(axis=1)
            upper_percentile = _df.drop(columns=["date"]).max(axis=1)
        else:
            raise ValueError(
                "Invalid percentile_type. Must be '25_75', '10_90', or 'min_max'."
            )

        # Create DataFrame for the area
        series = (
            "Min/Max"
            if percentile_type == "min_max"
            else (
                "25th/75th Percentiles"
                if percentile_type == "25_75"
                else "10th/90th Percentiles"
            )
        )
        area_data = pd.DataFrame(
            {
                "date": _df["date"],
                "lower": lower_percentile,
                "upper": upper_percentile,
                "Series": series,
            }
        )

        # Customize the x-axis
        alt_x = alt.X(
            "date:T",
            axis=alt.Axis(
                title=self.x_axis_title,
                titleFontSize=self.axis_title_font_size,
                format=self.date_format,
                labelFontSize=self.tick_font_size,
                labelAngle=self.x_ticks_angle,
                grid=False,
                labelAlign="center",  # Centers labels under their tick marks
            ),
        )

        # Customize the y-axis
        yticks = list(range(y_tick_min, y_tick_max + y_tick_step, y_tick_step))
        alt_y_scale = alt.Scale(domain=[yticks[0], yticks[-1]])
        alt_y_axis = alt.Axis(
            values=yticks,
            title=self.y_axis_title,
            titleFontSize=self.axis_title_font_size,
            labelFontSize=self.tick_font_size,
            titleAnchor="start",  # Puts y-axis title in bottom left corner of plot
            titleAngle=0,  # Makes y-axis title horizontal
            titleY=-10,  # Moves y-axis title up to the upper left corner of plot
            gridDash=[2, 2],
            gridColor="darkgray",
        )

        # Create the area chart
        alt_area_legend = alt.Legend(
            title=None,
            labelFontSize=self.axis_title_font_size,
            offset=1,  # Offset in pixels by which to displace the legend from the data rectangle and axes.
            symbolSize=300,  # Length of the variable’s stroke in the legend
            symbolStrokeWidth=5,  # Width of the variable’s stroke in the legend
            orient=legend_box_orient
            if legend_box_orient
            else self.legend_box_orient,  # Position of legend box in plot
            labelLimit=0,  # Ensures labels are not truncated
            fillColor="white",  # Background color of the legend box
            symbolType="square",  # Shape of the legend symbol
        )

        area_chart = (
            alt.Chart(area_data)
            .mark_area(
                opacity=0.3 if percentile_type == "25_75" else 0.15, color=area_color
            )
            .encode(
                x=alt_x,
                y=alt.Y("lower:Q", scale=alt_y_scale, axis=alt_y_axis),
                y2="upper:Q",
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(range=[area_color]),
                    legend=alt_area_legend,
                ),
            )
        )

        area_chart = area_chart.properties(width=self.width, height=self.height)

        if base_chart:
            chart = (base_chart + area_chart).resolve_scale(color="independent")
            return chart
        else:
            return area_chart


if __name__ == "__main__":
    pass
