"""MyPyTable represents a 2D table with column names
"""

import copy
import csv

# from mysklearn import myutils

#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
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

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        num_rows = len(self.data)
        num_cols = len(self.data[0])

        return num_rows, num_cols

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
        col_index = self.column_names.index(col_identifier)

        column = []
        for row in self.data:
            if row[col_index] or not include_missing_values:
                column.append(row[col_index])

        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                try:
                    self.data[i][j] = float(val)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for index in sorted(row_indexes_to_drop, reverse=True):
            self.data.pop(index)

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
        with open(filename, 'r') as file:
            reader = csv.reader(file)

            self.data = []
            for row in reader:
                self.data.append(row)
            self.column_names = self.data.pop(0)

            self.convert_to_numeric()

            return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline = '') as file:
            writer = csv.writer(file)

            writer.writerow(self.column_names)

            for row in self.data:
                writer.writerow(row)

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
        duplicates = []

        cols = []
        for column_name in key_column_names:
            cols.append(self.column_names.index(column_name))

        val_set = set()
        for index, row in enumerate(self.data):
            vals = []
            for col in cols:
                vals.append(row[col])

            if str(vals) in val_set:
                duplicates.append(index)
            else:
                val_set.add(str(vals))

        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for index, row in reversed(list(enumerate(self.data))):
            for val in row:
                if val == 'NA':
                    self.data.pop(index)
                    break

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.column_names.index(col_name)

        col_sum = 0
        count = 0
        nas = []
        for index, row in enumerate(self.data):
            val = row[col]
            if val == 'NA':
                nas.append(index)
            else:
                col_sum += val
                count += 1

        avg = col_sum / count

        for index in nas:
            self.data[index][col] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        summary_table_header = ["attribute", "min", "max", "mid", "avg", "median"]

        summary_data = []
        if self.data:
            for col_name in col_names:
                col_index = self.column_names.index(col_name)

                column = []
                col_sum = 0
                for row in self.data:
                    if row[col_index] != 'NA':
                        column.append(row[col_index])
                        col_sum += row[col_index]
                column = sorted(column)

                col_min = column[0]
                col_max = column[-1]

                middle_index = len(column) // 2
                if len(column) % 2 == 1:
                    median = column[middle_index]
                else:
                    median = (column[middle_index] + column[middle_index - 1]) / 2

                summary_row = [col_name, col_min, col_max, (col_min + col_max) / 2,
                               col_sum / len(column), median]
                summary_data.append(summary_row)

        return MyPyTable(summary_table_header, summary_data)


    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_data = []

        col_index1 = []
        col_index2 = []
        for key_column_name in key_column_names:
            col_index1.append(self.column_names.index(key_column_name))
            col_index2.append(other_table.column_names.index(key_column_name))

        for row1 in self.data:
            for row2 in other_table.data:
                matching = True
                for i in range(len(key_column_names)):
                    if row1[col_index1[i]] != row2[col_index2[i]]:
                        matching = False

                if matching:
                    new_row = copy.deepcopy(row1)
                    for i, val in enumerate(row2):
                        if i not in col_index2:
                            new_row.append(val)
                    new_data.append(new_row)

        # create header
        new_header = copy.deepcopy(self.column_names)
        for column_name in other_table.column_names:
            if column_name not in new_header:
                new_header.append(column_name)

        return MyPyTable(new_header, new_data)

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
        new_data = []

        col_index1 = []
        col_index2 = []
        for key_column_name in key_column_names:
            col_index1.append(self.column_names.index(key_column_name))
            col_index2.append(other_table.column_names.index(key_column_name))

        other_table_not_found = copy.deepcopy(other_table.data)
        def left_outer():
            for row1 in self.data:
                found = False
                for row2 in other_table.data:
                    matching = True
                    for i in range(len(key_column_names)):
                        if row1[col_index1[i]] != row2[col_index2[i]]:
                            matching = False

                    if matching:
                        new_row = copy.deepcopy(row1)
                        for i, val in enumerate(row2):
                            if i not in col_index2:
                                new_row.append(val)
                        new_data.append(new_row)
                        found = True

                        if row2 in other_table_not_found:
                            other_table_not_found.remove(row2)

                if not found:  # left outer
                    new_row = copy.deepcopy(row1)
                    for i in range(len(other_table.column_names) - len(key_column_names)):
                        new_row.append('NA')
                    new_data.append(new_row)

        left_outer()

        def right_outer():
            for row in other_table_not_found:
                new_row = []
                for col_name in self.column_names:
                    if col_name not in key_column_names:
                        new_row.append('NA')
                    elif col_name in other_table.column_names:
                        col_index = other_table.column_names.index(col_name)
                        new_row.append(row[col_index])
                for i, val in enumerate(row):
                    if other_table.column_names[i] not in self.column_names:
                        new_row.append(val)
                new_data.append(new_row)

        right_outer()

        def create_header():
            new_header = copy.deepcopy(self.column_names)
            for column_name in other_table.column_names:
                if column_name not in new_header:
                    new_header.append(column_name)
            return new_header

        new_header = create_header()

        return MyPyTable(new_header, new_data)

    def get_columns(self, col_identifiers, include_missing_values=True):
        """Return a new MyPyTable that is the columns from this MyPyTable with column_names in col_identifiers

        Args:
            col_identifiers(list of str): column names to get the columns of
            include_missing_values(bool): whether or not to include missing values in the new table

        Returns:
            MyPyTable: the new table with columns corresponding to col_identifiers
        """
        col_indices = [self.column_names.index(col_identifier) for col_identifier in col_identifiers]

        table = MyPyTable()
        table.column_names = col_identifiers
        for row in self.data:
            new_row = []
            for col_index in col_indices:
                if row[col_index] or not include_missing_values:
                    new_row.append(row[col_index])
            table.data.append(new_row)

        return table

    def add_column(self, col_name, column, index=None):
        """Adds a new column to the table.

        Args:
            col_name(str): the name of the new column
            column(list of obj): the values of the new column
            index(int): the index for where to add the column, defaults to the end of the table

        Returns:
            MyPyTable: the new table with columns corresponding to col_identifiers
        """
        if index is None:
            if self.data:
                index=len(self.data[0])
            else:
                index=0

        self.column_names.insert(index, col_name)
        for i, row in enumerate(self.data):
            row.insert(index, column[i])

    def get_column_index(self, col_name):
        return self.column_names.index(col_name)

    def group_by(self, col_name):
        col_index = self.get_column_index(col_name)
        partitions = []
        values = []
        for row in self.data:
            if row[col_index] not in values:
                values.append(row[col_index])
                partitions.append(MyPyTable(self.column_names, [row]))
            else:
                partitions[values.index(row[col_index])].data.append(row)

        return partitions

    def drop_column(self, col_name):
        col_index = self.get_column_index(col_name)

        if len(self.column_names) >0:
            self.column_names.pop(col_index)

        for row in self.data:
            row.pop(col_index)
