import os
import utils
from mysklearn.mypytable import MyPyTable

fpath = os.path.join("Data", "beer_reviews.csv")
table = MyPyTable().load_from_file(fpath)

print("loaded")

table.drop_column("review_taste")
table.drop_column("review_palate")
table.drop_column("review_profilename")
table.drop_column("review_aroma")
table.drop_column("review_time")
table.drop_column("review_appearance")

groups = table.group_by("beer_beerid")

print("grouped")

cleaned_table = MyPyTable()
cleaned_table.column_names = table.column_names
cleaned_table.column_names.remove("review_overall")
cleaned_table.column_names.append("rating")

for group in groups.values():
    col = group.get_column("review_overall")

    average = sum(col) / len(col)
    val = utils.high_low_discretizer(average)

    group.drop_column("review_overall")

    row = group.data[0]
    row.append(val)
    cleaned_table.data.append(row)

# cleaned_table.save_to_file("beer_reviews_cleaned.csv")
