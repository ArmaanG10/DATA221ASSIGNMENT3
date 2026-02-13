import pandas as pd
import matplotlib.pyplot as plt

# load crime dataset
crime_df = pd.read_csv('crime.csv')
violent_crimes = crime_df['ViolentCrimesPerPop']

# set up figure size
plt.figure(figsize=(12, 5))

# create histogram
plt.subplot(1, 2, 1)
plt.hist(violent_crimes.dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Violent Crimes Per Population')
plt.xlabel('Violent Crimes Per Population')
plt.ylabel('Frequency')

# create boxplot
plt.subplot(1, 2, 2)
plt.boxplot(violent_crimes.dropna(), vert=False, patch_artist=True)
plt.title('Box Plot of Violent Crimes Per Population')
plt.xlabel('Violent Crimes Per Population')
plt.ylabel('Distribution')

plt.tight_layout()
plt.show()

# COMMENTS FOR QUESTION 2:
# The histogram shows how the data values are spread out. It emphasizes the most common ranges of crime rates and shows the overall skewness of the dataset.
# The boxplot shows where the median sits inside the interquartile range, which represents the middle 50% of our data.
# The box plot also suggests the presence of outliers. These are usually represented as individual dots or circles sitting far outside the main "whiskers" of the plot.