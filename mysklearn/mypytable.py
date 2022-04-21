import copy
import csv
import random as rd
from statistics import mean
from statistics import median

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
    def __str__(self):
        spacer = " "
        head = self.column_names
        spaces = len(max(head))
        # for name in head:
        #     spaces= spaces + len(name)
        # spaces = spaces//len(head) + 4
        for i in range(spaces):
            spacer = spacer + "  "
        print(spacer.join(head))
        print()

        temp=""
        for j,i in enumerate(spacer):
            if j != len(spacer)-1:
                temp = temp + i
        spacer = temp
        for line in self.data:
            new_line = [str(a) for a in line]
            i = 0
            length =0
            for val in new_line:
                try:
                    float(val)
                    val = round(val,2)

                except:
                    val

                if i == 0:
                    print(val,end="")
                else:
                    length = len(new_line[i-1]) -1
                    temp = len(head[i-1]) + len(spacer) - length
                    for j in range(temp):
                        print(" ",end="")
                    print(val,end="")
                i = i+1
            print()

        return""

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


    def get_shape(self):
        """Computes the dimension of the table (N x M).
            Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        len(self.column_names)
        return len(self.data),len(self.column_names)

    def get_rows_with_val(self,col,val):
        """
        Returns: a mypytable with only the value in the requested row
        """

        mt = MyPyTable()

        for row in self.data:
            if row[col] == val:
                mt.data.append(row)
        return mt
    def how_many_vals_in_col (self,col):
        """
        """
        out = {}
        coll = self.get_column(col)
        temp=[]
        for val in coll:
            if not val in temp:
                temp .append(val)
                out[val]=0

        for val in temp :
            for row in coll:
                if row == val:
                    out[val] = out[val] + 1
        return out




    def get_row(self,index,delete=False):
        """Extracts a row from the list
        Args:
            index(int): A row in the column to be returned
            delete is true deletes row after returned
        Returns:
            the list obj of a row
        """
        x = self.data[index]
        if delete is True:
            self.drop_rows([index])
        return x


    def get_column(self, col_identifier, include_missing_values=True,delete_col = False):

        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
            delete_col(bool) True removes the column form the table after it is retuned
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        if type(col_identifier) == type(1):
            i = col_identifier
        else:
            i = self.column_names.index(col_identifier)
        temp = []

        if include_missing_values == True:
            for line in self.data:
                temp.append(line[i])
        else:
            for line in self.data:
                if not "NA" in line[i]:
                    temp.append(line[i])

        if delete_col == True:
            self.drop_column(col_identifier)

        return temp

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        i =0
        for list in self.data:
            j =0
            for val in list:
                try:
                   self.data[i][j] = float(val)
                except:
                    try:
                        self.data[i][j] = int(val)
                    except:
                        val = val
                j = j+1
            i = i+1


        pass

    def drop_column(self, col_index_to_drop):
        """Removes a column
        Args:
            col_index_to_drop(int): index of column to remove

        """

        if len(self.column_names) >0:
            self.column_names.pop(col_index_to_drop)
        i =0
        for line in self.data:
            self.data[i].pop(col_index_to_drop)
            i=i+1



        pass

    def add_column(self, new_col_data,col_name = "New Column",loc=-1):
        """Adds a new column to the data in my table
        Args:
            new_col_data(list of len of data): new data
            col_name(string): header for the new column
        """

        self.column_names.append(col_name)
        i=0
        for line in self.data:
            line.append(new_col_data[i])
            i = i + 1
        if self.data==[]:

            i=0
            while i < len(new_col_data):
                temp =[]
                temp.append(new_col_data[i])
                self.data.append(temp)
                i = i + 1
        pass


    def shuffle(self):
        """Shuffles the rows of the data
        """
        rd.shuffle(self.data)

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse = True)
        for drop in row_indexes_to_drop:
            self.data.pop(drop)
        pass
    def sort_on_column(self,column):
        """ sorts table on a given column
        Args:
            column(int) Column to sort on
        """
        x = self.data
        self.data = sorted(self.data,key=lambda x:x[column])

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
        with open(filename, 'r', encoding="utf8") as file:
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
            Used the csv documentation to help with this one
        """
        with open(filename, 'w', newline = '', encoding="utf8") as file:
            writer = csv.writer(file)

            writer.writerow(self.column_names)

            for row in self.data:
                writer.writerow(row)

        pass

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
        table = copy.deepcopy(self)
        dups =[]
        j =[]
        remove = []
        for name in table.column_names:
            if not name in key_column_names:
                remove.append(name)
        for name in remove:
            j.append(table.column_names.index(name))
            j.sort(reverse=True)
        for other in j:
            table.drop_column(other)

        temp =[]
        i=0
        for line in table.data:
            if not line in temp:
                temp.append(line)
            else:
                dups.append(i)
            i = i+1

        return dups

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").

        """
        self.data = [line for line in self.data if "NA" not in line]
        pass

    def remove_rows_with_values(self,value):
        """Remove rows from the table data that contain a missing value (the user picks ).
        Args: value what should be removed
        """
        self.data = [line for line in self.data if value not in line]
        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        self.convert_to_numeric()
        temp = self.get_column(col_name,include_missing_values=False)
        avg = mean(temp)
        i=0
        for line in self.data:
            j=0
            for val in line :
                if val == "NA":
                    self.data[i][j] = avg
                j = j+1
            i=i+1
        pass
    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        head =["attribute", "min", "max", "mid", "avg", "median"]

        data =[]
        if self.data != []:
            self.convert_to_numeric()
            self.remove_rows_with_missing_values()
            print()
            for col in col_names:
                line =[]
                temp = self.get_column(col)
                temp.sort()
                line .append(col)
                line.append(round(temp[0],3))
                line.append(round(temp[-1],3))
                other = (temp[-1]+temp[0])/2
                line.append(round(other,3))
                line.append(round(mean(temp),3))
                line.append(round(median(temp),3))
                data.append(line)
            return MyPyTable(head,data)
        else :
            return MyPyTable(head,[])



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


        new_head =[]
        data =[]
        newline =[]
        new_head = copy.deepcopy(self.column_names)
        for name in other_table.column_names:
            if not name in new_head:
                new_head.append(name)
        other = []
        me =[]
        tmp =[]
        i=0
        used =[]
        while i < (len(other_table.data)):
            used.append(0)
            i=i+1
        for val in key_column_names:
            me.append(self.column_names.index(val))
        for val in key_column_names:
            other.append(other_table.column_names.index(val))

        for val in key_column_names:
            tmp.append(new_head.index(val))

        i =0
        merge = False
        for line in self.data:
            home = True
            merge = False

            k=0
            for temp in other_table.data:
                shouldDie = True
                j =0
                for val in me:

                    if line[me[j]] == temp[other[j]]:
                        merge = True
                        shouldDie = True
                    else :
                        shouldDie = False

                        break
                    j= j+1

                if (merge is True and shouldDie is True) :
                    used[k] = 1
                    for con in line:
                        newline.append(con)
                    for tp in temp :
                        if not tp in newline:
                            newline.append(tp)

                    home = False
                    data.append(newline)
                    newline =[]

                k = k+1

            if home is True:
                for con in line:
                    newline.append(con)
                i =0
                while i <(len(temp) - len(key_column_names)):
                    newline.append("NA")
                    i=i+1

                data.append(newline)
                newline =[]
        i =0

        for val in used:
            newline = []
            if val ==0:
                line = other_table.data[i]
                j=0
                for id in new_head:
                    if id in other_table.column_names:
                        newline.append(line[other_table.column_names.index(id)])
                    else:
                        newline.append("NA")
                data.append(newline)

            i = i +1

        return MyPyTable(new_head,data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        temp = self.perform_full_outer_join(other_table,key_column_names)
        temp.remove_rows_with_missing_values()


    def get_column_index(self, col_name):
        return self.column_names.index(col_name)

    def group_by(self, col_name):
        col_index = self.get_column_index(col_name)

        partitions = {}
        for row in self.data:
            value = row[col_index]

            if value in partitions:
                partitions[value].data.append(row)
            else:
                partitions[value] = MyPyTable(self.column_names, [row])
        return partitions

    def drop_column(self, col_name):
        col_index = self.get_column_index(col_name)

        if col_index:
            if len(self.column_names) >0:
                self.column_names.pop(col_index)

            for row in self.data:
                row.pop(col_index)
