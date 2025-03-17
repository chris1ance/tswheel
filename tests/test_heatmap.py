"""Tests for HeatmapPlotter's make_heatmap method in tswheel/viz/heatmap.py."""

import os
import pandas as pd
import pytest
import altair as alt
from tswheel.viz.heatmap import HeatmapPlotter


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
        pdf_path = os.path.join(output_dir, "heatmaps.pdf")
        combined_chart.save(pdf_path)

        # Save combined chart as PNG
        png_path = os.path.join(output_dir, "heatmaps.png")
        combined_chart.save(png_path)


class TestHeatmapPlotter:
    @pytest.fixture
    def plotter(self):
        """Create a HeatmapPlotter instance."""
        return HeatmapPlotter(width=600, height=400)

    @pytest.fixture
    def correlation_data(self):
        """Create sample correlation matrix data."""
        return pd.DataFrame(
            {
                "A": [1.0, 0.7, 0.3, -0.2],
                "B": [0.7, 1.0, 0.5, 0.1],
                "C": [0.3, 0.5, 1.0, 0.8],
                "D": [-0.2, 0.1, 0.8, 1.0],
            },
            index=["A", "B", "C", "D"],
        )

    @pytest.fixture
    def small_data(self):
        """Create a small sample heatmap data."""
        return pd.DataFrame(
            {
                "X": [10, 20, 30],
                "Y": [15, 25, 35],
                "Z": [5, 15, 25],
            },
            index=["P", "Q", "R"],
        )

    def test_basic_heatmap(self, plotter, correlation_data):
        """Test creating a basic heatmap with default settings."""
        chart = plotter.make_heatmap(correlation_data)

        # Add to chart store for visual inspection
        ChartStore.charts.append(chart)

        # Check that the chart is an Altair LayerChart (not Chart)
        assert isinstance(chart, alt.LayerChart)

        # Check dimensions
        assert chart.width == 600
        assert chart.height == 400

        # Check that there are two layers (heatmap + text)
        assert len(chart.layer) == 2

        # Verify encoding channels in first layer (heatmap rectangles)
        assert chart.layer[0].encoding.x.shorthand == "Index:O"
        assert chart.layer[0].encoding.y.shorthand == "Columns:O"
        assert chart.layer[0].encoding.color.shorthand == "Value:Q"

        # Verify encoding channels in second layer (text values)
        assert chart.layer[1].encoding.x.shorthand == "Index:O"
        assert chart.layer[1].encoding.y.shorthand == "Columns:O"
        assert chart.layer[1].encoding.text.shorthand == "Value:Q"

    def test_custom_styling(self, plotter, correlation_data):
        """Test heatmap with custom styling options."""
        chart = plotter.make_heatmap(
            data=correlation_data,
            text_fontsize=16,
            title="Correlation Matrix",
            x_axis_title="Variables X",
            y_axis_title="Variables Y",
            x_ticks_angle=45,
        )

        ChartStore.charts.append(chart)

        # Check title
        assert chart.title.text == "Correlation Matrix"

        # Check axis titles - use dict access instead of property access
        assert chart.layer[0].encoding.x.to_dict()["title"] == "Variables X"
        assert chart.layer[0].encoding.y.to_dict()["title"] == "Variables Y"

        # Check x-axis label angle - use to_dict() for accessing axis properties
        assert chart.layer[0].encoding.x.to_dict()["axis"]["labelAngle"] == 45

        # Check text font size
        assert chart.layer[1].mark.fontSize == 16

    def test_small_dataset(self, plotter, small_data):
        """Test with a small dataset to ensure correct rendering."""
        chart = plotter.make_heatmap(
            data=small_data,
            title="Small Heatmap Example",
        )

        ChartStore.charts.append(chart)

        # Basic assertions - check for LayerChart instead of Chart
        assert isinstance(chart, alt.LayerChart)
        assert chart.title.text == "Small Heatmap Example"

        # Verify the chart has data
        assert chart.layer[0].data is not None

        # Verify the original data dimensions match what we'd expect
        melted_data = small_data.reset_index(names="Index").melt(
            id_vars="Index", var_name="Columns", value_name="Value"
        )
        assert len(melted_data) == 9  # 3 rows × 3 columns
        assert set(melted_data["Index"].unique()) == set(["P", "Q", "R"])
        assert set(melted_data["Columns"].unique()) == set(["X", "Y", "Z"])

    def test_width_height_setters(self, plotter):
        """Test that width and height properties can be modified."""
        # Initial values
        assert plotter.width == 600
        assert plotter.height == 400

        # Change dimensions
        plotter.width = 800
        plotter.height = 600

        # Check new values
        assert plotter.width == 800
        assert plotter.height == 600

        # Create chart with new dimensions
        chart = plotter.make_heatmap(
            data=pd.DataFrame(
                {"A": [1.0, 0.5], "B": [0.5, 1.0]},
                index=["A", "B"],
            )
        )

        ChartStore.charts.append(chart)

        # Verify the dimensions were applied
        assert chart.width == 800
        assert chart.height == 600
