"""Utilities for Altair line plots."""

from typing import Union, Literal
import pandas as pd
from fredapi import Fred
import altair as alt
from functools import lru_cache

pd.set_option("mode.copy_on_write", True)


class LinePlotter:
    def __init__(self, fred_api_key: str | None = None):
        self.fred_api_key = fred_api_key
        self.recession_bars_plot = None

    @lru_cache(maxsize=None)
    def get_recession_periods(
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
        recession_spans = self.get_recession_periods(
            start_date=start_date,
            end_date=end_date,
            nber_based=nber_based,
        )

        recession_bars_plot = (
            alt.Chart(recession_spans)
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
            _df = df.reset_index()

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
        date_format: str = "%Y",
        title: str = "",
        title_font_size: int = 24,
        x_axis_title: str | None = None,
        y_axis_title: str | None = None,
        axis_title_font_size: int = 20,
        tick_font_size: int = 18,
        x_ticks_angle: int = 0,
        width: int = 800,
        height: int = 400,
        legend_box_orient: Literal[
            "none",
            "left",
            "right",
            "top",
            "bottom",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ] = "top-left",
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
        date_format : str, optional
            Format string for x-axis date labels. Default: "%Y" (year only).
        title : str, optional
            Chart title. Default: "" (no title).
        title_font_size : int, optional
            Font size for chart title in pixels. Default: 24.
        x_axis_title : str|None, optional
            X-axis title. If None, no title is shown.
        y_axis_title : str|None, optional
            Y-axis title. If None, no title is shown.
        axis_title_font_size : int, optional
            Font size for axis titles in pixels. Default: 20.
        tick_font_size : int, optional
            Font size for axis tick labels in pixels. Default: 18.
        x_ticks_angle : int, optional
            Rotation angle for x-axis tick labels in degrees. Default: 0.
        width : int, optional
            Chart width in pixels. Default: 800.
        height : int, optional
            Chart height in pixels. Default: 400.
        legend_box_orient : Literal["none", "left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"], optional
            Position of the legend box in the plot. Default: "top-left".

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
                format=date_format,
                labelFontSize=tick_font_size,
                labelAngle=x_ticks_angle,
            ),
            scale=alt.Scale(nice=True),
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
                labelFontSize=tick_font_size,
                titleAnchor="start",  # Puts y-axis title in bottom left corner of plot
                titleAngle=0,  # Makes y-axis title horizontal
                titleY=-10,  # Moves y-axis title up to the upper left corner of plot
            ),
        )

        # Customize series legend
        alt_legend = alt.Legend(
            title=None,
            labelFontSize=axis_title_font_size,
            offset=3,  # Offset in pixels by which to displace the legend from the data rectangle and axes.
            symbolSize=300,  # Length of the variable’s stroke in the legend
            symbolStrokeWidth=20,  # Width of the variable’s stroke in the legend
            orient=legend_box_orient,  # Position of legend box in plot
            labelLimit=0,  # Ensures labels are not truncated
            strokeColor="black",  # Color of border around the legend
            fillColor="white",  # Background color of the legend box
        )

        # Series colors and series legend
        series_names = list(series_colors.keys())
        series_colors = list(series_colors.values())
        alt_scale = alt.Scale(domain=series_names, range=series_colors)
        alt_color = alt.Color("Series:N", scale=alt_scale, legend=alt_legend)

        # Make chart
        # clip=True: Clip any marks (e.g. lines or points) falling outside specified scale domain
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

        chart = (
            chart.properties(width=width, height=height, title=alt_title)
            .configure_axisX(grid=False)
            .configure_axisY(gridDash=[2, 2], gridColor="darkgray")
        )

        return chart


if __name__ == "__main__":
    pass
