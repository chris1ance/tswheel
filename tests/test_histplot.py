"""Tests for make_histogram method in tswheel/distributions.py."""

import os
import numpy as np
import pandas as pd
import pytest
import altair as alt
from tswheel.distplot import DistributionPlotter


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
        pdf_path = os.path.join(output_dir, "histograms.pdf")
        combined_chart.save(pdf_path)

        # Save combined chart as PNG
        png_path = os.path.join(output_dir, "histograms.png")
        combined_chart.save(png_path)


class TestDistributionPlotter:
    @pytest.fixture
    def plotter(self):
        """Create a DistributionPlotter instance."""
        return DistributionPlotter(width=600, height=300)

    @pytest.fixture
    def normal_data(self):
        """Generate a DataFrame with normally distributed data."""
        np.random.seed(42)
        data = np.random.normal(loc=5.0, scale=1.5, size=1000)
        return pd.DataFrame({"values": data})

    @pytest.fixture
    def bimodal_data(self):
        """Generate a DataFrame with bimodally distributed data."""
        np.random.seed(43)
        data1 = np.random.normal(loc=3.0, scale=1.0, size=500)
        data2 = np.random.normal(loc=7.0, scale=1.0, size=500)
        data = np.concatenate([data1, data2])
        return pd.DataFrame({"values": data})

    @pytest.fixture
    def uniform_data(self):
        """Generate a DataFrame with uniformly distributed data."""
        np.random.seed(44)
        data = np.random.uniform(low=0, high=10, size=1000)
        return pd.DataFrame({"values": data})

    def test_basic_histogram(self, plotter, normal_data):
        """Test creating a basic histogram with default settings."""
        chart = plotter.make_histogram(
            data=normal_data,
            value_column="values",
            bin_step=0.5,
            title="Basic Histogram",
        )

        assert isinstance(chart, alt.Chart)
        assert chart.width == 600
        assert chart.height == 300

        # Store chart for use in test_export_combined_charts
        self.basic_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)

    def test_custom_styling(self, plotter, normal_data):
        """Test histogram with custom styling options."""
        # Set font sizes using setter methods
        plotter.set_title_font_size(20)
        plotter.set_axis_title_font_size(16)
        plotter.set_tick_font_size(14)

        chart = plotter.make_histogram(
            data=normal_data,
            value_column="values",
            bin_step=0.5,
            color="steelblue",
            opacity=0.7,
            title="Custom Styled Histogram",
            x_axis_title="Values",
            y_axis_title="Frequency",
            x_tick_decimal_places=2,
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for use in test_export_combined_charts
        self.custom_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)

        # Reset plotter font sizes to defaults for subsequent tests
        plotter.set_title_font_size(24)
        plotter.set_axis_title_font_size(20)
        plotter.set_tick_font_size(18)

    def test_with_mean_line(self, plotter, normal_data):
        """Test histogram with a mean line."""
        chart = plotter.make_histogram(
            data=normal_data,
            value_column="values",
            bin_step=0.5,
            title="Histogram with Mean Line",
            show_mean_line=True,
            mean_line_color="darkred",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for use in test_export_combined_charts
        self.mean_line_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)

    def test_with_median_line(self, plotter, bimodal_data):
        """Test histogram with a median line."""
        chart = plotter.make_histogram(
            data=bimodal_data,
            value_column="values",
            bin_step=0.5,
            title="Histogram with Median Line",
            show_median_line=True,
            median_line_color="darkgreen",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for use in test_export_combined_charts
        self.median_line_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)

    def test_with_both_lines(self, plotter, bimodal_data):
        """Test histogram with both mean and median lines."""
        chart = plotter.make_histogram(
            data=bimodal_data,
            value_column="values",
            bin_step=0.3,
            title="Histogram with Mean and Median Lines",
            show_mean_line=True,
            show_median_line=True,
            mean_line_color="purple",
            median_line_color="teal",
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for use in test_export_combined_charts
        self.both_lines_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)

    def test_uniform_distribution(self, plotter, uniform_data):
        """Test histogram with uniformly distributed data."""
        chart = plotter.make_histogram(
            data=uniform_data,
            value_column="values",
            bin_step=0.5,
            title="Uniform Distribution Histogram",
            color="goldenrod",
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for use in test_export_combined_charts
        self.uniform_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)

    def test_width_height_setters(self, plotter, normal_data):
        """Test the width and height setter methods."""
        plotter.set_width(800)
        plotter.set_height(400)

        chart = plotter.make_histogram(
            data=normal_data,
            value_column="values",
            bin_step=0.5,
            title="Custom Size Histogram",
        )

        assert chart.width == 800
        assert chart.height == 400

        # Store chart for use in test_export_combined_charts
        self.custom_size_chart = chart

        # Add to global chart collection for session-level export
        ChartStore.charts.append(chart)
