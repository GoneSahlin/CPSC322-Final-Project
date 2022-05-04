"""Creates pie charts displaying our classifier's bias towards different beer styles
"""

import os
from matplotlib import pyplot as plt
import pickle

from mysklearn.mypytable import MyPyTable


infile = open("classifier.p", "rb")
clf = pickle.load(infile)
infile.close()

fpath = os.path.join("Data", "joined_data.csv")
table = MyPyTable().load_from_file(fpath)

table = table.get_columns(['beer_style', 'beer_abv', 'rating', 'brewery_country', 'brewery_rating'])
table.remove_rows_with_missing_values()

X = table.get_columns(['beer_style', 'beer_abv', 'brewery_country', 'brewery_rating']).data
y = table.get_column('rating')

y_predicted = clf.predict(X)

table.add_column("y_predicted", y_predicted)
groups = table.group_by("beer_style")

i = 1
fig = plt.figure()
for style in groups.keys():
    plt.subplot(6,7,i)
    group = groups[style]
    counts = {"low":0, "high":0}
    for row in group.data:
        counts[row[-1]] += 1

    plt.pie(counts.values(), labels=counts.keys(), colors=['teal', 'darkorange'])
    plt.title(style, size=10)
    i+=1

fig.tight_layout(pad=1)
plt.savefig("Other/style_bias.png")
