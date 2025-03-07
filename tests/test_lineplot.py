"""Tests for LinePlotter in tswheel/lineplot.py."""

import os
import numpy as np
import pandas as pd
import pytest
import altair as alt
from tswheel.lineplot import LinePlotter


# Create a class to store charts across test instances
class ChartStore:
    charts = []


# Module-level fixture to collect and export all charts at the end of the test session
@pytest.fixture(scope="session", autouse=True)
def export_all_charts_after_tests(request):
    """Export all charts to the tests/output directory after all tests have run."""
    # This part runs before any tests
    yield
    # This part runs after all tests have finished

    if ChartStore.charts:
        # Create output directory in the tests folder
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Concatenate charts vertically
        combined_chart = alt.vconcat(*ChartStore.charts, spacing=30).resolve_scale(
            color="independent"
        )

        # Save combined chart as PDF
        pdf_path = os.path.join(output_dir, "combined_lineplots.pdf")
        combined_chart.save(pdf_path)

        # Save combined chart as PNG
        png_path = os.path.join(output_dir, "combined_lineplots.png")
        combined_chart.save(png_path)


class TestLinePlotter:
    @pytest.fixture
    def plotter(self):
        """Create a LinePlotter instance."""
        return LinePlotter(fred_api_key=None, width=600, height=300)

    @pytest.fixture
    def time_series_data(self):
        """Generate a DataFrame with time series data."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        series1 = np.cumsum(np.random.normal(0, 1, 100))
        series2 = np.cumsum(np.random.normal(0, 1.5, 100))
        return pd.DataFrame({"date": dates, "Series1": series1, "Series2": series2})

    @pytest.fixture
    def single_series_data(self):
        """Generate a DataFrame with a single time series."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        series1 = np.cumsum(np.random.normal(0, 1, 100))
        return pd.DataFrame({"date": dates, "Series1": series1})

    @pytest.fixture
    def multiple_series_data(self):
        """Generate a DataFrame with multiple time series."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({"date": dates})

        for i in range(1, 6):
            data[f"Series{i}"] = np.cumsum(np.random.normal(0, i / 2, 100))

        return data

    def test_elicit_date_column(self, plotter, time_series_data):
        """Test the elicit_date_column method."""
        # Test with 'date' column already present
        df_with_date = time_series_data.copy()
        result_df = plotter.elicit_date_column(df_with_date)
        assert "date" in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])

        # Test with DatetimeIndex
        df_with_index = time_series_data.copy()
        df_with_index = df_with_index.set_index("date")
        result_df = plotter.elicit_date_column(df_with_index)
        assert "date" in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])

        # Test with 'Date' column (capitalized)
        df_with_cap_date = time_series_data.copy()
        df_with_cap_date = df_with_cap_date.rename(columns={"date": "Date"})
        result_df = plotter.elicit_date_column(df_with_cap_date)
        assert "date" in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])

    def test_width_height_setters(self, plotter):
        """Test the width and height setter methods."""
        plotter.set_width(800)
        plotter.set_height(400)

        assert plotter.width == 800
        assert plotter.height == 400

    def test_make_zero_hline_plot(self, plotter):
        """Test the make_zero_hline_plot static method."""
        yticks = [-10, -5, 0, 5, 10]
        chart = LinePlotter.make_zero_hline_plot(yticks)

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_basic_line_chart(self, plotter, time_series_data):
        """Test creating a basic line chart."""
        chart = plotter.make_line_chart(
            data=time_series_data,
            y_tick_min=-10,
            y_tick_max=10,
            y_tick_step=5,
            title="Basic Line Chart",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)
        assert chart.width == 600
        assert chart.height == 300

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_line_chart_custom_styling(self, plotter, time_series_data):
        """Test line chart with custom styling options."""
        chart = plotter.make_line_chart(
            data=time_series_data,
            y_tick_min=-10,
            y_tick_max=10,
            y_tick_step=5,
            series_colors={"Series1": "steelblue", "Series2": "darkred"},
            title="Custom Styled Line Chart",
            x_axis_title="Date",
            y_axis_title="Value",
            legend_title="Variables",
            title_font_size=20,
            axis_title_font_size=16,
            tick_font_size=14,
            date_format="%b %Y",
            x_ticks_angle=45,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_line_chart_legend_positions(self, plotter, time_series_data):
        """Test line chart with different legend positions."""
        chart = plotter.make_line_chart(
            data=time_series_data,
            y_tick_min=-10,
            y_tick_max=10,
            y_tick_step=5,
            title="Line Chart with Right Legend",
            legend_box_orient="right",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

        chart = plotter.make_line_chart(
            data=time_series_data,
            y_tick_min=-10,
            y_tick_max=10,
            y_tick_step=5,
            title="Line Chart with Top Legend",
            legend_box_orient="top",
            legend_direction="horizontal",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_area_chart_25_75(self, plotter, multiple_series_data):
        """Test area chart with 25th to 75th percentile range."""
        chart = plotter.make_area_chart(
            data=multiple_series_data,
            area_color="steelblue",
            y_tick_min=-10,
            y_tick_max=20,
            y_tick_step=5,
            title="Area Chart (25th-75th Percentiles)",
            percentile_type="25_75",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_area_chart_10_90(self, plotter, multiple_series_data):
        """Test area chart with 10th to 90th percentile range."""
        chart = plotter.make_area_chart(
            data=multiple_series_data,
            area_color="darkred",
            y_tick_min=-10,
            y_tick_max=20,
            y_tick_step=5,
            title="Area Chart (10th-90th Percentiles)",
            percentile_type="10_90",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_area_chart_min_max(self, plotter, multiple_series_data):
        """Test area chart with min to max range."""
        chart = plotter.make_area_chart(
            data=multiple_series_data,
            area_color="darkgreen",
            y_tick_min=-10,
            y_tick_max=20,
            y_tick_step=5,
            title="Area Chart (Min-Max Range)",
            percentile_type="min_max",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_area_chart_with_median(self, plotter, multiple_series_data):
        """Test area chart with median line."""
        chart = plotter.make_area_chart(
            data=multiple_series_data,
            area_color="purple",
            y_tick_min=-10,
            y_tick_max=20,
            y_tick_step=5,
            title="Area Chart with Median Line",
            percentile_type="25_75",
            center_line="median",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_area_chart_with_mean(self, plotter, multiple_series_data):
        """Test area chart with mean line."""
        chart = plotter.make_area_chart(
            data=multiple_series_data,
            area_color="teal",
            y_tick_min=-10,
            y_tick_max=20,
            y_tick_step=5,
            title="Area Chart with Mean Line",
            percentile_type="25_75",
            center_line="mean",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)

    def test_area_chart_with_all_lines(self, plotter, multiple_series_data):
        """Test area chart with all individual series lines."""
        series_colors = {f"Series{i}": f"hsl({i * 60}, 70%, 50%)" for i in range(1, 6)}

        chart = plotter.make_area_chart(
            data=multiple_series_data,
            area_color="lightgray",
            y_tick_min=-10,
            y_tick_max=20,
            y_tick_step=5,
            title="Area Chart with All Series Lines",
            percentile_type="25_75",
            center_line="all",
            series_colors=series_colors,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for export
        ChartStore.charts.append(chart)
