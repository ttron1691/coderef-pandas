# Code Reference for Pandas
## Install Packages
```Bash
python pip install pandas
```
## Include Packages
```Python
import pandas as pd
```
## Series
A series is the basic one-dimensional data structure in pandas. The constainer consists of an axis label and a numpy array structure.

In general, a pandas series can be created using the corresponding constructor
```Python
class pandas.Series(data=None,
  		    index=None,
		    dtype=None,
		    name=None,
	            copy=None,
		    fastpath=False)
```
Examples
```Python
s1 = pd.Series([1, 2, 3])
s2 = 0
```
The output of a series looks as follows
```
0    1
1    2
2    3
dtype: int64
```
## Data Frame creation
A data frame is the basic two-dimensional tabular data structure in pandas. The container consists of an index for rows and columns, respectively.

The constructor of a pandas data frame is given as follows
```Python
class pandas.DataFrame(data=None,
		       index=None,
		       columns=None,
		       dtype=None,
		       copy=None)
```

```Python
df = pd.DataFrame(data={"name": ["A", "B", "C"],
                        "weight": [5, -2, 3],
                        "abs": [1.25, 9.24, -0.12]})
```
This results in 
```
	name	weight	abs
0	A	5	1.25
1	B	-2	9.24
2	C	3	-0.12
```
### Joining Data Frames
Data Frames can be joined with "merge". The syntax is as follows
```Python
DataFrame.merge(right,
                how='inner',
                on=None,
                left_on=None,
 		right_on=None,
		left_index=False,
		right_index=False,
		sort=False,
		suffixes=('_x', '_y'),
		copy=None,
		indicator=False,
		validate=None)
```
Example for a merge
```Python
df_merged = df1.merge(df2, how="left", on="a")
```
## Input and Output
```Python
# CSV format
df.load_csv()
df.to_csv()

# Excel format
df.load_excel()
df.to_excel()
```
## References
The pandas documentation can be found on: [https://pandas.pydata.org/docs/index.html](https://pandas.pydata.org/docs/index.html)
