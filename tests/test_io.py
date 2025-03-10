import json
import pickle
import pytest
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

from tswheel.io import IO


class TestIO:
    """Test suite for the IO class."""

    @pytest.fixture
    def temp_dir(self):
        """Fixture providing a temporary directory for test files."""
        with TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_dict(self) -> Dict[str, Any]:
        """Fixture providing a sample dictionary for testing."""
        return {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Fixture providing a sample DataFrame for testing."""
        return pd.DataFrame(
            {"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]}
        )

    def test_init_with_io_paths(self, temp_dir, sample_dict):
        """Test initialization with input and output paths."""
        # Create test directories and files
        input_dir = temp_dir / "data"
        input_dir.mkdir(exist_ok=True)
        input_file = input_dir / "input.json"

        # Write test data to file
        with open(input_file, "w") as f:
            json.dump(sample_dict, f)

        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Initialize with actual paths to existing files
        io = IO(
            input_dir=input_dir,
            input_filename="input.json",
            output_dir=output_dir,
            output_filename="output.json",
        )

        # Check that paths are properly set
        assert io.input_dir == input_dir
        assert io.input_filename == Path("input.json")
        assert io.output_dir == output_dir
        assert io.output_filename == Path("output.json")
        assert io.input_path == input_file
        assert io.output_path == output_dir / "output.json"

    def test_init_with_tmp_paths(self, temp_dir):
        """Test initialization with temporary paths."""
        # Create test directories
        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Test with temporary output directory
        io = IO(
            output_dir=output_dir,
            output_filename="output.json",
            make_tmp_output_dir=True,
        )
        assert io.output_dir == Path(f"{output_dir}_tmp")
        assert io.output_path == Path(f"{output_dir}_tmp/output.json")

        # Test with temporary output file
        io = IO(
            output_dir=output_dir,
            output_filename="output.json",
            make_tmp_output_file=True,
        )
        assert io.output_filename == Path("output_tmp.json")
        assert io.output_path == output_dir / "output_tmp.json"

    def test_init_errors(self, temp_dir):
        """Test initialization errors."""
        # Test error when make_tmp_output_file is True but no output_filename
        with pytest.raises(ValueError):
            IO(make_tmp_output_file=True)

        # Test error when input path does not exist
        non_existent_dir = temp_dir / "non_existent_dir"
        with pytest.raises(FileNotFoundError):
            IO(input_dir=non_existent_dir, input_filename="non_existent.json")

    def test_pickle_io(self, temp_dir, sample_dict):
        """Test reading and writing pickle files."""
        # Create a pickle file
        pickle_path = temp_dir / "test.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(sample_dict, f)

        # Test read_pickle_file
        result = IO.read_pickle_file(pickle_path)
        assert result == sample_dict

        # Test write_pickle_file
        output_path = temp_dir / "output.pkl"
        IO.write_pickle_file(sample_dict, output_path)
        with open(output_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == sample_dict

    def test_json_io(self, temp_dir, sample_dict):
        """Test reading and writing JSON files."""
        # Create a JSON file
        json_path = temp_dir / "test.json"
        with open(json_path, "w") as f:
            json.dump(sample_dict, f)

        # Test read_json_file
        result = IO.read_json_file(json_path)
        assert result == sample_dict

        # Test write_json_file
        output_path = temp_dir / "output.json"
        IO.write_json_file(sample_dict, output_path)
        with open(output_path, "r") as f:
            loaded = json.load(f)
        assert loaded == sample_dict

        # Test write_json_file with non-dict input
        with pytest.raises(TypeError):
            IO.write_json_file(["not a dict"], output_path)

    def test_read_input(self, temp_dir, sample_dict, sample_df):
        """Test the read_input method with different file types."""
        # Create test files
        pickle_path = temp_dir / "test.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(sample_dict, f)

        json_path = temp_dir / "test.json"
        with open(json_path, "w") as f:
            json.dump(sample_dict, f)

        csv_path = temp_dir / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        excel_path = temp_dir / "test.xlsx"
        sample_df.to_excel(excel_path, index=False)

        # Test reading each file type
        io_pkl = IO(input_dir=temp_dir, input_filename="test.pkl")
        result_pkl = io_pkl.read_input()
        assert result_pkl == sample_dict

        io_json = IO(input_dir=temp_dir, input_filename="test.json")
        result_json = io_json.read_input()
        assert result_json == sample_dict

        io_csv = IO(input_dir=temp_dir, input_filename="test.csv")
        result_csv = io_csv.read_input()
        pd.testing.assert_frame_equal(result_csv, sample_df)

        io_excel = IO(input_dir=temp_dir, input_filename="test.xlsx")
        result_excel = io_excel.read_input()
        pd.testing.assert_frame_equal(result_excel, sample_df)

        # Test error for unsupported extension
        with open(temp_dir / "test.txt", "w") as f:
            f.write("This is a test")
        io_unsupported = IO(input_dir=temp_dir, input_filename="test.txt")
        with pytest.raises(ValueError):
            io_unsupported.read_input()

    def test_write_output(self, temp_dir, sample_dict, sample_df):
        """Test the write_output method with different file types."""
        # Test writing pickle
        io_pkl = IO(output_dir=temp_dir, output_filename="output.pkl")
        io_pkl.write_output(sample_dict)
        assert (temp_dir / "output.pkl").exists()
        with open(temp_dir / "output.pkl", "rb") as f:
            loaded_pkl = pickle.load(f)
        assert loaded_pkl == sample_dict

        # Test writing JSON
        io_json = IO(output_dir=temp_dir, output_filename="output.json")
        io_json.write_output(sample_dict)
        assert (temp_dir / "output.json").exists()
        with open(temp_dir / "output.json", "r") as f:
            loaded_json = json.load(f)
        assert loaded_json == sample_dict

        # Test writing CSV
        io_csv = IO(output_dir=temp_dir, output_filename="output.csv")
        io_csv.write_output(sample_df)
        assert (temp_dir / "output.csv").exists()
        # Read CSV without creating index column
        loaded_csv = pd.read_csv(temp_dir / "output.csv")
        # The loaded CSV has an extra index column, so we need to drop it for comparison
        if "Unnamed: 0" in loaded_csv.columns:
            loaded_csv = loaded_csv.drop(columns=["Unnamed: 0"])
        pd.testing.assert_frame_equal(loaded_csv, sample_df)

        # Test writing Excel
        io_excel = IO(output_dir=temp_dir, output_filename="output.xlsx")
        io_excel.write_output(sample_df)
        assert (temp_dir / "output.xlsx").exists()
        loaded_excel = pd.read_excel(temp_dir / "output.xlsx")
        # The loaded Excel may have an extra index column, so we need to drop it for comparison
        if "Unnamed: 0" in loaded_excel.columns:
            loaded_excel = loaded_excel.drop(columns=["Unnamed: 0"])
        pd.testing.assert_frame_equal(loaded_excel, sample_df)

        # Test error for no output path
        io_no_output = IO()
        with pytest.raises(FileNotFoundError):
            io_no_output.write_output(sample_dict)

        # Test error for unsupported extension
        io_unsupported = IO(output_dir=temp_dir, output_filename="output.txt")
        with pytest.raises(ValueError):
            io_unsupported.write_output(sample_dict)

    def test_directory_creation(self, temp_dir):
        """Test that output directories are created as needed."""
        nested_dir = temp_dir / "nested" / "directory"
        io = IO(output_dir=nested_dir, output_filename="output.json")

        # Directory should not exist yet
        assert not nested_dir.exists()

        # Writing should create the directory
        io.write_output({"test": "data"})

        # Directory should now exist
        assert nested_dir.exists()
        assert (nested_dir / "output.json").exists()
