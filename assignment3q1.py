import pandas as pd

# Q1
# load crime dataset
crime_df = pd.read_csv('crime.csv')

# focus on specific column
violent_crimes = crime_df['ViolentCrimesPerPop']

# compute statistical measures
mean_val = violent_crimes.mean()
median_val = violent_crimes.median()
std_val = violent_crimes.std()
min_val = violent_crimes.min()
max_val = violent_crimes.max()

print("--- Question 1 Statistics ---")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Standard Deviation: {std_val}")
print(f"Minimum: {min_val}")
print(f"Maximum: {max_val}\n")

# COMMENTS FOR QUESTION 1:
# Looking at the results the mean (aprox. 0.441) is higher than the median (aprox. 0.39).
# Because the mean is larger, the distribution is right skewed and not symmetric. This means a few areas with really high violent crime rates are making the average go up.
# If there are extreme values in the dataset the mean is the statistic that gets affected the most.
# This is because calculating the mean takes the mathematical average considering all values, while median ignores any extreme values (high or low) and takes the middle value(s).