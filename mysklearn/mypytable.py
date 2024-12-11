"""
Programmers: Michael D'Arcy-Evans and Isabel Tilles
Class: CPSC322-01, Fall 2024
Final Project
12/6/2024
We attempted the bonus.

Description: A table class consisting of a header list and a 2d list of data along with numerous standard table functions
"""

import copy
import csv
from tabulate import tabulate


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []

        if isinstance(col_identifier, str):
            col_index = self.column_names.index(col_identifier)
        else:
            if col_identifier < len(self.column_names):
                col_index = col_identifier
            else:
                print("Attempted to access an invalid index.")
                return None

        for row in self.data:
            if include_missing_values:
                col.append(row[col_index])
            else:
                if row[col_index] != "NA":
                    col.append(row[col_index])

        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for index, element in enumerate(row):
                try:
                    row[index] = float(element)
                except ValueError:
                    row[index] = element

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_data = []
        for index, row in enumerate(self.data):
            if index not in row_indexes_to_drop:
                new_data.append(row)
        self.data = new_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, "r", encoding="utf-8") as fin:
            for line in csv.reader(fin, skipinitialspace=True):
                if self.column_names == []:
                    self.column_names = line
                else:
                    self.data.append(line)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w", encoding="utf-8") as fout:
            file_writer = csv.writer(fout)
            file_writer.writerow(self.column_names)
            file_writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_index_values = [
            self.column_names.index(element) for element in key_column_names
        ]
        temp_list = set()
        duplicate_list = []

        if len(key_index_values) < 2:
            for element in key_index_values:
                for index, row in enumerate(self.data):
                    if row[element] in temp_list:
                        duplicate_list.append(index)
                    else:
                        temp_list.add(row[element])
        else:
            for index, row in enumerate(self.data):
                try:
                    if (
                        row[key_index_values[0]],
                        row[key_index_values[1]],
                    ) in temp_list:
                        duplicate_list.append(index)
                    else:
                        temp_list.add(
                            (row[key_index_values[0]], row[key_index_values[1]])
                        )
                except IndexError:
                    break
        return duplicate_list

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        list_to_remove = []
        for index, row in enumerate(self.data):
            for element in row:
                if element in ["NA", ""]:
                    list_to_remove.append(index)
        self.drop_rows(list_to_remove)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        col_average = sum(self.get_column(col_name, False)) / len(
            self.get_column(col_name, False)
        )
        for index, row in enumerate(self.data):
            if row[col_index] == "NA":
                self.data[index][col_index] = col_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        data = []
        for col_name in col_names:
            row = []
            column = self.get_column(col_name, False)
            if column:
                row.append(col_name)
                row.append(min(column))
                row.append(max(column))
                row.append((min(column) + max(column)) / 2)
                row.append(sum(column) / len(column))
                if len(column) % 2 != 0:
                    row.append(sorted(column)[int((len(column)) / 2)])
                else:
                    row.append(
                        (
                            sorted(column)[int((len(column)) / 2)]
                            + sorted(column)[int((len(column)) / 2) - 1]
                        )
                        / 2
                    )
                data.append(row)

        return MyPyTable(header, data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_header = self.column_names + [
            col_name
            for col_name in other_table.column_names
            if col_name not in self.column_names
        ]
        joined_data = []
        other_key_dict = {}

        self_key_index_values = [
            self.column_names.index(element) for element in key_column_names
        ]

        other_key_index_values = [
            other_table.column_names.index(element) for element in key_column_names
        ]

        for row in other_table.data:
            key = tuple(row[i] for i in other_key_index_values)
            if key not in other_key_dict:
                other_key_dict[key] = []
            other_key_dict[key].append(row)

        for row in self.data:
            key = tuple(row[i] for i in self_key_index_values)
            if key in other_key_dict:
                for other_row in other_key_dict[key]:
                    new_row = list(row)
                    for col_name in other_table.column_names:
                        if col_name not in self.column_names:
                            new_row.append(
                                other_row[other_table.column_names.index(col_name)]
                            )
                    joined_data.append(new_row)

        return MyPyTable(joined_header, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_data = []
        new_row = []
        other_key_dict = {}
        matched_keys = set()

        joined_header = self.column_names + [
            col_name
            for col_name in other_table.column_names
            if col_name not in self.column_names
        ]

        self_key_index_values = [
            self.column_names.index(element) for element in key_column_names
        ]
        other_key_index_values = [
            other_table.column_names.index(element) for element in key_column_names
        ]

        for row in other_table.data:
            key = tuple(row[i] for i in other_key_index_values)
            if key not in other_key_dict:
                other_key_dict[key] = []
            other_key_dict[key].append(row)

        for row in self.data:
            key = tuple(row[i] for i in self_key_index_values)
            if key in other_key_dict:
                matched_keys.add(key)
                for other_row in other_key_dict[key]:
                    new_row = list(row) + [
                        other_row[other_table.column_names.index(col_name)]
                        for col_name in other_table.column_names
                        if col_name not in self.column_names
                    ]
                    joined_data.append(new_row)

        for row in self.data:
            key = tuple(row[i] for i in self_key_index_values)
            if key not in matched_keys:
                new_row = list(row) + ["NA"] * (
                    len(other_table.column_names) - len(self_key_index_values)
                )
                joined_data.append(new_row)

        for row in other_table.data:
            key = tuple(row[i] for i in other_key_index_values)
            if key not in matched_keys:
                new_row = ["NA"] * len(joined_header)
                for item in row:
                    if other_table.column_names[row.index(item)] in joined_header:
                        new_row[
                            joined_header.index(
                                other_table.column_names[row.index(item)]
                            )
                        ] = item
                joined_data.append(new_row)

        return MyPyTable(joined_header, joined_data)
