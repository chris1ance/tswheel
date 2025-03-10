from pathlib import Path
from typing import Optional, Union, Any
import pickle
import pandas as pd
import json


class IO:
    """
    A class to manage input/output operations for various file formats.

    Supports reading and writing of:
    - Pickle files (.pkl)
    - JSON files (.json)
    - CSV files (.csv)
    - Excel files (.xlsx)
    - PNG images (.png)
    - PDF documents (.pdf)

    Attributes:
        input_dir (Optional[Path]): Directory containing input files
        input_filename (Optional[Path]): Name of the input file
        output_dir (Optional[Path]): Directory for output files
        output_filename (Optional[Path]): Name of the output file
        make_tmp_output_dir (bool): Whether to create a temporary output directory
        make_tmp_output_file (bool): Whether to create a temporary output file
        input_path (Optional[Path]): Full path to the input file (derived from input_dir and input_filename)
        output_path (Optional[Path]): Full path to the output file (derived from output_dir and output_filename)
    """

    SUPPORTED_EXTENSIONS = {".pkl", ".json", ".csv", ".xlsx", ".png", ".pdf"}

    def __init__(
        self,
        input_dir: Union[str, Path, None] = None,
        input_filename: Union[str, Path, None] = None,
        output_dir: Union[str, Path, None] = None,
        output_filename: Union[str, Path, None] = None,
        make_tmp_output_dir: bool = False,
        make_tmp_output_file: bool = False,
    ) -> None:
        """
        Initialize the IOManager with input and output paths.

        Creates Path objects for input and output locations, automatically constructing
        full paths when both directory and filename components are provided. When temporary
        file or directory options are enabled, the paths are modified by appending '_tmp'.

        Args:
            input_dir (Union[str, Path, None]): Directory containing input files. If None,
                input_path is set to None, even if input_filename is provided.
            input_filename (Union[str, Path, None]): Name of the input file. Required with
                input_dir to create a valid input_path.
            output_dir (Union[str, Path, None]): Directory for output files. If None but
                output_filename is provided, current working directory is used.
            output_filename (Union[str, Path, None]): Name of the output file. If None,
                output_path is set to None.
            make_tmp_output_dir (bool): Whether to create a temporary output directory by
                appending '_tmp' to the output_dir name.
            make_tmp_output_file (bool): Whether to create a temporary output file by
                appending '_tmp' to the output_filename stem.

        Raises:
            ValueError: If output_filename is not specified but make_tmp_output_file is True,
                since a temporary filename cannot be created without a base filename.
            FileNotFoundError: If input_path is constructed (both input_dir and input_filename
                are provided) but the resulting file doesn't exist on the filesystem.

        Examples:
            >>> io = IOManager(
            ...     input_dir="data/raw",
            ...     input_filename="dataset.csv",
            ...     output_dir="data/processed",
            ...     output_filename="cleaned_data.csv"
            ... )

            >>> # Create temporary output file
            >>> io_tmp = IOManager(
            ...     output_dir="results",
            ...     output_filename="report.json",
            ...     make_tmp_output_file=True
            ... )  # Will create results/report_tmp.json
        """
        # Convert input parameters to Path objects if they are provided as strings.
        self.input_dir: Optional[Path] = Path(input_dir) if input_dir else None
        self.input_filename: Optional[Path] = (
            Path(input_filename) if input_filename else None
        )
        self.output_dir: Optional[Path] = Path(output_dir) if output_dir else None
        self.output_filename: Optional[Path] = (
            Path(output_filename) if output_filename else None
        )
        self.make_tmp_output_dir: bool = make_tmp_output_dir
        self.make_tmp_output_file: bool = make_tmp_output_file

        # Adjust output_dir if temporary output directory is requested.
        if self.make_tmp_output_dir:
            self.output_dir = Path(f"{self.output_dir}_tmp")

        # Adjust output_filename if temporary output file is requested.
        if self.make_tmp_output_file:
            if self.output_filename:
                self.output_filename = Path(
                    f"{self.output_filename.stem}_tmp{self.output_filename.suffix}"
                )
            else:
                raise ValueError(
                    "output_filename not specified but make_tmp_output_file is True."
                )

        # Combine output directory and filename to form the full output path, if possible.
        if self.output_filename:
            self.output_path: Optional[Path] = (
                self.output_dir / self.output_filename
                if self.output_dir
                else Path.cwd() / self.output_filename
            )
        else:
            self.output_path = None

        # Similarly, combine input directory and filename to form the full input path, if possible.
        if self.input_dir and self.input_filename:
            self.input_path: Optional[Path] = self.input_dir / self.input_filename
        else:
            self.input_path = None

    @staticmethod
    def read_pickle_file(input_path: Union[str, Path]) -> Any:
        """
        Read a pickle file and return its contents.

        Args:
            input_path (Union[str, Path]): Path to the pickle file to read.

        Returns:
            Any: The unpickled contents of the file.
        """
        with open(input_path, "rb") as file:
            contents = pickle.load(file)
        return contents

    @staticmethod
    def write_pickle_file(output: Any, output_path: Union[str, Path]) -> None:
        """
        Write data to a pickle file.

        Args:
            output (Any): Data to pickle and write to file.
            output_path (Union[str, Path]): Path where the pickle file will be written.
        """
        with open(output_path, "wb") as file:
            pickle.dump(output, file)

    @staticmethod
    def read_json_file(input_path: Union[str, Path]) -> dict:
        """
        Read a JSON file and return its contents as a dictionary.

        Args:
            input_path (Union[str, Path]): Path to the JSON file to read.

        Returns:
            dict: The parsed JSON contents.
        """
        with open(input_path, "r") as file:
            contents = json.load(file)
        return contents

    @staticmethod
    def write_json_file(output: dict, output_path: Union[str, Path]) -> None:
        """
        Write a dictionary to a JSON file.

        Args:
            output (dict): Dictionary to write as JSON.
            output_path (Union[str, Path]): Path where the JSON file will be written.

        Raises:
            TypeError: If output is not a dictionary.
        """
        if not isinstance(output, dict):
            raise TypeError("Output must be a dictionary for JSON files")
        with open(output_path, "w") as file:
            json.dump(
                output, file, indent=4
            )  # indent=4 makes the file human-readable with nice formatting

    def read_input(self, **kwargs) -> Any:
        """
        Read the input file based on its extension.

        Args:
            **kwargs: Additional arguments passed to the underlying read function

        Returns:
            Any: The contents of the input file. For CSV and Excel files, returns a pandas DataFrame.
                For pickle and JSON files, returns the deserialized data.

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If input path is not set or if input file does not exist
        """
        if not self.input_path:
            raise FileNotFoundError("Input path is not set")
        elif self.input_path and not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} does not exist.")

        suffix = self.input_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {suffix}")

        if suffix == ".pkl":
            return self.read_pickle_file(self.input_path)
        if suffix == ".json":
            return self.read_json_file(self.input_path)
        if suffix == ".csv":
            return pd.read_csv(self.input_path, **kwargs)
        if suffix == ".xlsx":
            return pd.read_excel(self.input_path, **kwargs)

    def write_output(self, output: Any, **kwargs) -> None:
        """
        Write data to the output file based on its extension.

        Args:
            output (Any): Data to write. For CSV and Excel, should be a pandas DataFrame.
                For JSON, should be a dictionary. For pickle, can be any serializable object.
                For PNG and PDF, should be an image object with a save method.
            **kwargs: Additional arguments passed to the underlying write function

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If output path is not set
            TypeError: If output is not the correct type for the specified file format
        """
        if not self.output_path:
            raise FileNotFoundError("Output path is not set")

        # Create output directory only if it's specified
        if self.output_dir:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = self.output_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {suffix}")

        if suffix == ".pkl":
            self.write_pickle_file(output, self.output_path)
        elif suffix == ".json":
            self.write_json_file(output, self.output_path)
        elif suffix == ".csv":
            output.to_csv(self.output_path, **kwargs)
        elif suffix == ".xlsx":
            output.to_excel(self.output_path, **kwargs)
        elif suffix == ".png":
            output.save(self.output_path, ppi=150)
        elif suffix == ".pdf":
            output.save(self.output_path)
