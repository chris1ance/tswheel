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

    @pytest.fixture
    def faceted_data(self):
        """Generate a DataFrame with multiple groups and facets."""
        np.random.seed(47)

        # Create three groups with different distributions
        group_a = np.random.normal(loc=5.0, scale=1.0, size=200)
        group_b = np.random.normal(loc=7.0, scale=1.5, size=200)
        group_c = np.random.normal(loc=3.0, scale=0.8, size=200)

        # Create two facet categories with different means
        facet_1 = np.concatenate([group_a, group_b, group_c])
        facet_2 = np.concatenate(
            [group_a + 2, group_b - 1, group_c + 1]
        )  # Shift values

        # Combine the data into a DataFrame
        df = pd.DataFrame(
            {
                "values": np.concatenate([facet_1, facet_2]),
                "group": np.concatenate(
                    [
                        np.repeat("Group A", 200),
                        np.repeat("Group B", 200),
                        np.repeat("Group C", 200),
                        np.repeat("Group A", 200),
                        np.repeat("Group B", 200),
                        np.repeat("Group C", 200),
                    ]
                ),
                "facet": np.concatenate(
                    [
                        np.repeat("Category X", 600),
                        np.repeat("Category Y", 600),
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

    def test_basic_faceted_boxplot(self, plotter, faceted_data):
        """Test boxplot with facets across columns."""
        chart = plotter.make_boxplot(
            data=faceted_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Boxplot with Facets",
        )

        # Faceted chart should be a FacetChart
        assert isinstance(chart, alt.FacetChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_custom_order(self, plotter, faceted_data):
        """Test boxplot with facets and custom facet order."""
        # Custom order for facets (reversed)
        facet_order = ["Category Y", "Category X"]

        chart = plotter.make_boxplot(
            data=faceted_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            facet_order=facet_order,
            title="Boxplot with Custom Facet Order",
        )

        assert isinstance(chart, alt.FacetChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_with_group_order(self, plotter, faceted_data):
        """Test boxplot with facets and custom group order."""
        # Custom order for groups
        group_order = ["Group C", "Group A", "Group B"]

        chart = plotter.make_boxplot(
            data=faceted_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            custom_order=group_order,
            title="Boxplot with Facets and Custom Group Order",
        )

        assert isinstance(chart, alt.FacetChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_styling(self, plotter, faceted_data):
        """Test faceted boxplot with custom styling."""
        # Custom colors for groups
        color_mapping = {
            "Group A": "steelblue",
            "Group B": "darkgreen",
            "Group C": "firebrick",
        }

        chart = plotter.make_boxplot(
            data=faceted_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            series_colors=color_mapping,
            median_color="black",
            box_width=40,
            title="Styled Faceted Boxplot",
            x_axis_title="Group",
            y_axis_title="Value",
            x_ticks_angle=30,
        )

        assert isinstance(chart, alt.FacetChart)

        # Store chart for global export
        ChartStore.charts.append(chart)

    @pytest.fixture
    def faceted_mixed_sign_data(self):
        """Generate a DataFrame with multiple groups and facets with values crossing zero."""
        np.random.seed(48)

        # Create three groups with different distributions, all crossing zero
        group_a = np.random.normal(loc=0.0, scale=1.0, size=200)  # centered at zero
        group_b = np.random.normal(
            loc=2.0, scale=2.0, size=200
        )  # mostly positive but some negative
        group_c = np.random.normal(
            loc=-2.0, scale=2.0, size=200
        )  # mostly negative but some positive

        # Create two facet categories with different shifts
        facet_1 = np.concatenate([group_a, group_b, group_c])
        facet_2 = np.concatenate(
            [group_a - 1, group_b + 1, group_c - 2]
        )  # Different shifts

        # Combine the data into a DataFrame
        df = pd.DataFrame(
            {
                "values": np.concatenate([facet_1, facet_2]),
                "group": np.concatenate(
                    [
                        np.repeat("Group A", 200),
                        np.repeat("Group B", 200),
                        np.repeat("Group C", 200),
                        np.repeat("Group A", 200),
                        np.repeat("Group B", 200),
                        np.repeat("Group C", 200),
                    ]
                ),
                "facet": np.concatenate(
                    [
                        np.repeat("Category X", 600),
                        np.repeat("Category Y", 600),
                    ]
                ),
            }
        )

        return df

    def test_faceted_boxplot_with_zero_line_yticks(
        self, plotter, faceted_mixed_sign_data
    ):
        """Test faceted boxplot with data spanning zero using yticks list."""
        # Create a faceted boxplot with explicit y-ticks list that spans zero
        chart = plotter.make_boxplot(
            data=faceted_mixed_sign_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Faceted Boxplot with Zero Line (yticks)",
            y_tick_min=-4,
            y_tick_max=4,
            y_tick_step=1,  # Explicitly creates yticks list crossing zero
        )

        assert isinstance(chart, alt.FacetChart)

        # Check the original chart has layers (boxplot + zero line)
        # This is a bit tricky with faceted charts, but we can check the JSON spec
        chart_dict = chart.to_dict()
        # The zero line should be included in the specification
        assert "layer" in chart_dict["spec"]

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_with_zero_line_minmax(
        self, plotter, faceted_mixed_sign_data
    ):
        """Test faceted boxplot with data spanning zero using min/max values without step."""
        # Create a faceted boxplot with min/max spanning zero but no explicit tick step
        chart = plotter.make_boxplot(
            data=faceted_mixed_sign_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Faceted Boxplot with Zero Line (min/max)",
            y_tick_min=-5,
            y_tick_max=5,  # Min/max span zero with no step
        )

        assert isinstance(chart, alt.FacetChart)

        # Check that the chart spec includes layers
        chart_dict = chart.to_dict()
        assert "layer" in chart_dict["spec"]

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_custom_zero_line(self, plotter, faceted_mixed_sign_data):
        """Test faceted boxplot with custom zero line styling."""
        # Get a reference to the make_zero_hline_chart method to verify it's called
        # with custom parameters
        original_make_zero_hline = plotter.make_zero_hline_chart

        # Create a mock to replace the original method and customize zero line
        def custom_zero_hline(*args, **kwargs):
            # Call the original method with custom styling
            return original_make_zero_hline(
                *args,
                line_color="red",  # Custom zero line color
                line_size=2,  # Custom line size
                **kwargs,
            )

        # Replace the method temporarily
        plotter.make_zero_hline_chart = custom_zero_hline

        try:
            # Create the faceted boxplot
            chart = plotter.make_boxplot(
                data=faceted_mixed_sign_data,
                value_column="values",
                group_column="group",
                facet_column="facet",
                title="Faceted Boxplot with Custom Zero Line",
                y_tick_min=-4,
                y_tick_max=4,
                y_tick_step=2,
            )

            assert isinstance(chart, alt.FacetChart)

            # Check the chart specification
            chart_dict = chart.to_dict()
            assert "layer" in chart_dict["spec"]

            # Store chart for global export
            ChartStore.charts.append(chart)

        finally:
            # Restore the original method
            plotter.make_zero_hline_chart = original_make_zero_hline

    def test_faceted_boxplot_asymmetric_yticks(self, plotter, faceted_mixed_sign_data):
        """Test faceted boxplot with asymmetric yticks around zero."""
        # Create a faceted boxplot with asymmetric y-axis range spanning zero
        chart = plotter.make_boxplot(
            data=faceted_mixed_sign_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Faceted Boxplot with Asymmetric Y-ticks",
            y_tick_min=-2,  # Smaller negative range
            y_tick_max=6,  # Larger positive range
            y_tick_step=1,  # Still crosses zero
        )

        assert isinstance(chart, alt.FacetChart)

        # Check the chart specification has layers
        chart_dict = chart.to_dict()
        assert "layer" in chart_dict["spec"]

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_mixed_boxplot_some_zero_crossing(
        self, plotter, faceted_data, faceted_mixed_sign_data
    ):
        """Test complex case with mixed data where some facets cross zero and others don't."""
        # Create a combined dataset where one facet has all positive values and another crosses zero
        combined_data = pd.concat(
            [
                faceted_data[
                    faceted_data["facet"] == "Category X"
                ],  # Positive values only
                faceted_mixed_sign_data[
                    faceted_mixed_sign_data["facet"] == "Category Y"
                ],  # Crosses zero
            ]
        )

        # Set a y-axis range that covers both facets
        chart = plotter.make_boxplot(
            data=combined_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Mixed Facets with Different Zero Crossing",
            y_tick_min=-4,
            y_tick_max=8,
            y_tick_step=2,
        )

        assert isinstance(chart, alt.FacetChart)

        # The zero line should be present since the overall y-axis range crosses zero
        chart_dict = chart.to_dict()
        assert "layer" in chart_dict["spec"]

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_with_columns(self, plotter):
        """Test boxplot with facets using the columns parameter to limit columns in the layout."""
        # Create multi-facet dataset with 4 facet categories
        np.random.seed(49)

        # Create four facet categories with different values
        n_per_group = 100
        facet_categories = ["A", "B", "C", "D"]
        groups = ["Group 1", "Group 2", "Group 3"]

        data_list = []
        for i, facet in enumerate(facet_categories):
            for j, group in enumerate(groups):
                # Create slightly different distributions for each combination
                values = np.random.normal(
                    loc=3.0 + i - j, scale=0.5 + 0.2 * j, size=n_per_group
                )

                df = pd.DataFrame(
                    {
                        "values": values,
                        "group": np.repeat(group, n_per_group),
                        "facet": np.repeat(f"Category {facet}", n_per_group),
                    }
                )

                data_list.append(df)

        # Combine all the data
        combined_data = pd.concat(data_list, ignore_index=True)

        # Create a faceted chart with 2 columns layout
        chart = plotter.make_boxplot(
            data=combined_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Faceted Boxplot with 2 Columns Layout",
            columns=2,  # Limit to 2 columns in the grid
        )

        assert isinstance(chart, alt.FacetChart)

        # Check that the columns parameter was passed
        chart_dict = chart.to_dict()
        assert chart_dict.get("columns") == 2

        # Store chart for global export
        ChartStore.charts.append(chart)

    def test_faceted_boxplot_with_columns_and_zero_line(self, plotter):
        """Test boxplot with facets in a 2-column layout and data crossing zero to show zero line."""
        # Create multi-facet dataset with 4 facet categories
        np.random.seed(50)

        # Create four facet categories with data that crosses zero
        n_per_group = 100
        facet_categories = ["A", "B", "C", "D"]
        groups = ["Group 1", "Group 2", "Group 3"]

        data_list = []
        for i, facet in enumerate(facet_categories):
            for j, group in enumerate(groups):
                # Create distributions that cross zero by centering around -1, 0, 1, 2
                # for different facets, ensuring some data points cross zero
                values = np.random.normal(
                    loc=i - 1.5,  # Centers from -1.5 to +2.5
                    scale=1.0 + 0.2 * j,  # Add more variance by group
                    size=n_per_group,
                )

                df = pd.DataFrame(
                    {
                        "values": values,
                        "group": np.repeat(group, n_per_group),
                        "facet": np.repeat(f"Category {facet}", n_per_group),
                    }
                )

                data_list.append(df)

        # Combine all the data
        combined_data = pd.concat(data_list, ignore_index=True)

        # Create a faceted chart with 2 columns layout
        chart = plotter.make_boxplot(
            data=combined_data,
            value_column="values",
            group_column="group",
            facet_column="facet",
            title="Faceted Boxplot with 2 Columns and Zero Line",
            columns=2,  # Limit to 2 columns in the grid
            y_tick_min=-4,  # Set y-axis range to ensure it crosses zero
            y_tick_max=4,
            y_tick_step=1,
        )

        assert isinstance(chart, alt.FacetChart)

        # Check that the columns parameter was passed
        chart_dict = chart.to_dict()
        assert chart_dict.get("columns") == 2

        # Check that the zero line was added
        assert "layer" in chart_dict["spec"]

        # Store chart for global export
        ChartStore.charts.append(chart)
