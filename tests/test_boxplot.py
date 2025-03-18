"""Tests for make_boxplot method in tswheel/distributions.py."""

import os
import numpy as np
import pandas as pd
import pytest
import altair as alt
from tswheel.viz.distplot import DistributionPlotter


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
        pdf_path = os.path.join(output_dir, "boxplots.pdf")
        combined_chart.save(pdf_path)

        # Save combined chart as PNG
        png_path = os.path.join(output_dir, "boxplots.png")
        combined_chart.save(png_path)


class TestBoxPlot:
    @pytest.fixture
    def plotter(self):
        """Create a DistributionPlotter instance."""
        return DistributionPlotter(width=600, height=300)

    @pytest.fixture
    def single_group_data(self):
        """Generate a DataFrame with a single group of data."""
        np.random.seed(42)
        data = np.random.normal(loc=5.0, scale=1.5, size=1000)
        return pd.DataFrame({"values": data})

    @pytest.fixture
    def mixed_sign_data(self):
        """Generate a DataFrame with values crossing zero (both positive and negative)."""
        np.random.seed(44)
        data = np.random.normal(loc=0.0, scale=3.0, size=1000)
        return pd.DataFrame({"values": data})

    @pytest.fixture
    def negative_data(self):
        """Generate a DataFrame with only negative values."""
        np.random.seed(45)
        data = np.random.normal(loc=-5.0, scale=1.5, size=1000)
        return pd.DataFrame({"values": data})

    @pytest.fixture
    def mixed_sign_multi_group_data(self):
        """Generate a DataFrame with multiple groups with values crossing zero."""
        np.random.seed(46)

        # Create three groups with different distributions, all crossing zero
        group_a = np.random.normal(loc=0.0, scale=1.0, size=200)  # centered at zero
        group_b = np.random.normal(
            loc=2.0, scale=2.0, size=200
        )  # mostly positive but some negative
        group_c = np.random.normal(
            loc=-2.0, scale=2.0, size=200
        )  # mostly negative but some positive

        # Combine the data into a DataFrame
        df = pd.DataFrame(
            {
                "values": np.concatenate([group_a, group_b, group_c]),
                "group": np.concatenate(
                    [
                        np.repeat("Group A", 200),
                        np.repeat("Group B", 200),
                        np.repeat("Group C", 200),
                    ]
                ),
            }
        )

        return df

    @pytest.fixture
    def multi_group_data(self):
        """Generate a DataFrame with multiple groups of data."""
        np.random.seed(43)

        # Create three groups with different distributions
        group_a = np.random.normal(loc=5.0, scale=1.0, size=200)
        group_b = np.random.normal(loc=7.0, scale=1.5, size=200)
        group_c = np.random.normal(loc=3.0, scale=0.8, size=200)

        # Combine the data into a DataFrame
        df = pd.DataFrame(
            {
                "values": np.concatenate([group_a, group_b, group_c]),
                "group": np.concatenate(
                    [
                        np.repeat("Group A", 200),
                        np.repeat("Group B", 200),
                        np.repeat("Group C", 200),
                    ]
                ),
            }
        )

        return df

    def test_basic_boxplot(self, plotter, single_group_data):
        """Test creating a basic boxplot with default settings."""
        # Add a dummy group column for single boxplot
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Basic Boxplot",
        )

        assert isinstance(chart, alt.Chart)
        assert chart.width == 600
        assert chart.height == 300

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_custom_styling(self, plotter, single_group_data):
        """Test boxplot with custom styling options."""
        # Add a dummy group column for single boxplot
        data = single_group_data.copy()
        data["group"] = "All Data"

        # Set font sizes using setter methods
        plotter.set_title_font_size(20)
        plotter.set_axis_title_font_size(16)
        plotter.set_tick_font_size(14)

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            series_colors="greens",
            title="Custom Styled Boxplot",
            x_axis_title="",
            y_axis_title="Values",
            median_color="darkred",  # Custom median color
            box_width=40,
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

        # Reset plotter font sizes to defaults for subsequent tests
        plotter.set_title_font_size(24)
        plotter.set_axis_title_font_size(20)
        plotter.set_tick_font_size(18)

    def test_default_median_color(self, plotter, single_group_data):
        """Test boxplot with default median color (black)."""
        # Add a dummy group column for single boxplot
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Default Median Color",
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_no_outliers(self, plotter, single_group_data):
        """Test boxplot with outliers hidden."""
        # Add a dummy group column for single boxplot
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot without Outliers",
            show_outliers=False,
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_grouped_boxplot(self, plotter, multi_group_data):
        """Test boxplot with grouped data."""
        chart = plotter.make_boxplot(
            data=multi_group_data,
            value_column="values",
            group_column="group",
            title="Grouped Boxplot",
            x_axis_title="Group",
            y_axis_title="Values",
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_grouped_boxplot_custom_order(self, plotter, multi_group_data):
        """Test boxplot with custom group order."""
        custom_order = ["Group C", "Group A", "Group B"]  # Rearranged order

        chart = plotter.make_boxplot(
            data=multi_group_data,
            value_column="values",
            group_column="group",
            title="Grouped Boxplot with Custom Order",
            custom_order=custom_order,
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_custom_color_mapping(self, plotter, multi_group_data):
        """Test boxplot with custom color mapping for groups."""
        color_mapping = {
            "Group A": "steelblue",
            "Group B": "darkgreen",
            "Group C": "firebrick",
        }

        chart = plotter.make_boxplot(
            data=multi_group_data,
            value_column="values",
            group_column="group",
            series_colors=color_mapping,
            title="Boxplot with Custom Colors",
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_color_scheme(self, plotter, multi_group_data):
        """Test boxplot with a color scheme."""
        chart = plotter.make_boxplot(
            data=multi_group_data,
            value_column="values",
            group_column="group",
            series_colors="category10",
            title="Boxplot with Color Scheme",
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_custom_y_axis_ticks(self, plotter, single_group_data):
        """Test boxplot with custom y-axis tick values."""
        # Add a dummy group column for single boxplot
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Custom Y-Axis Ticks",
            y_tick_min=0,
            y_tick_max=10,
            y_tick_step=2,
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_y_axis_with_zero_line(self, plotter, mixed_sign_data):
        """Test boxplot with y-axis range spanning both positive and negative values (should show zero line)."""
        # Add a dummy group column
        data = mixed_sign_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Zero Line",
            y_tick_min=-6,
            y_tick_max=6,
            y_tick_step=2,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_y_axis_only_positive(self, plotter, single_group_data):
        """Test boxplot with y-axis range containing only positive values (no zero line)."""
        # Add a dummy group column
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Only Positive Values",
            y_tick_min=2,
            y_tick_max=10,
            y_tick_step=2,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_y_axis_only_negative(self, plotter, negative_data):
        """Test boxplot with y-axis range containing only negative values (no zero line)."""
        # Add a dummy group column
        data = negative_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Only Negative Values",
            y_tick_min=-10,
            y_tick_max=-2,
            y_tick_step=2,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_min_max_spans_zero_no_step(self, plotter, mixed_sign_data):
        """Test boxplot with min/max spanning zero but no explicit tick step (should add zero line)."""
        # Add a dummy group column
        data = mixed_sign_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Min/Max Spanning Zero (No Step)",
            y_tick_min=-5,
            y_tick_max=5,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_min_max_no_zero_span(self, plotter, single_group_data):
        """Test boxplot with min/max not spanning zero and no explicit tick step (no zero line)."""
        # Add a dummy group column
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Boxplot with Min/Max Not Spanning Zero",
            y_tick_min=2,
            y_tick_max=8,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_multi_group_with_zero_line(self, plotter, mixed_sign_multi_group_data):
        """Test multi-group boxplot with data spanning zero (should show zero line)."""
        chart = plotter.make_boxplot(
            data=mixed_sign_multi_group_data,
            value_column="values",
            group_column="group",
            title="Multi-Group Boxplot with Zero Line",
            y_tick_min=-6,
            y_tick_max=6,
            y_tick_step=2,
        )

        assert isinstance(chart, alt.Chart) or isinstance(chart, alt.LayerChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_width_height_setters(self, plotter, single_group_data):
        """Test the width and height setter methods with boxplot."""
        plotter.set_width(800)
        plotter.set_height(400)

        # Add a dummy group column for single boxplot
        data = single_group_data.copy()
        data["group"] = "All Data"

        chart = plotter.make_boxplot(
            data=data,
            value_column="values",
            group_column="group",
            title="Custom Size Boxplot",
        )

        assert chart.width == 800
        assert chart.height == 400

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_x_ticks_angle(self, plotter, multi_group_data):
        """Test boxplot with rotated x-axis tick labels."""
        chart = plotter.make_boxplot(
            data=multi_group_data,
            value_column="values",
            group_column="group",
            title="Boxplot with Rotated X-Axis Labels",
            x_ticks_angle=45,
        )

        assert isinstance(chart, alt.Chart)

        # Store chart for global export
        ChartStore.charts.append(chart)
