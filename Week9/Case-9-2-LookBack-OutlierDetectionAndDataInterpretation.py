"""
Reference:
https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/ 

"""

"""
Detecting Outliers Using Boxplot
Python code for boxplot is:

"""
import matplotlib.pyplot as plt

sample= [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
plt.boxplot(sample, vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample')
plt.show()

"""
Detecting Outliers using the Z-scores
Criteria: any data point whose Z-score falls out of 3rd standard deviation is an outlier treatment.

How to Handle Outliers
Detecting Outliers with Z-scores
Steps
loop through all the data points and compute the Z-score using the formula (Xi-mean)/std.
define a threshold value of 3 and mark the datapoints whose absolute value of Z-score is greater than the threshold as outliers.
"""
import numpy as np
outliers = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers# Driver code
sample_outliers = detect_outliers_zscore(sample)
print("Outliers from Z-scores method: ", sample_outliers)
#The above code outputs: Outliers from Z-scores method: [101]

"""
Next Case:
https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/

Detecting Outliers using the Inter Quantile Range(IQR)
Detecting outliers with iqr
IQR to detect Outliners
Criteria: data points that lie 1.5 times of IQR above Q3 and below Q1 are outliers. This shows in detail about outlier treatment in Python.

Steps
Sort the dataset in ascending order
calculate the 1st and 3rd quartiles(Q1, Q3)
compute IQR=Q3-Q1
compute lower bound = (Q1–1.5*IQR), upper bound = (Q3+1.5*IQR)
loop through the values of the dataset and check for those who fall below the lower bound and above the upper bound and mark them as outlier treatment in python
"""
outliers = []
def detect_outliers_iqr(data):
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code
sample_outliers = detect_outliers_iqr(sample)
print("Outliers from IQR method: ", sample_outliers)

######################## ends here

# Trimming for i in sample_outliers:     
a = np.delete(sample, np.where(sample==i)) 
print(a) 
print(len(sample), len(a))

# Computing 10th, 90th percentiles and replacing the outlier treatment in python
tenth_percentile = np.percentile(sample, 10) 
ninetieth_percentile = np.percentile(sample, 90) 
# print(tenth_percentile, ninetieth_percentile)b = 
np.where(sample<tenth_percentile, tenth_percentile, sample) 
b = np.where(b>ninetieth_percentile, ninetieth_percentile, b) 
# print("Sample:", sample) 
print("New array:",b)


#Step 3: Mean/Median Imputation
#As the mean value is highly influenced by the outlier treatment, it is advised to replace the outliers with the median value.
median = np.median(sample)
# Replace with median for i in sample_outliers:     
c = np.where(sample==i, 14, sample) 
print("Sample: ", sample) 
print("New array: ",c) 
print(x.dtype)

#Step 5: Visualizing the Data after Treating the Outlier
plt.boxplot(c, vert=False)
plt.title("Boxplot of the sample after treating the outliers")
plt.xlabel("Sample")

##############################  ends here