# Code Reference for Pandas
## Install Packages
```Bash
python -m pip install pandas
```
## Include Packages
```Python
import pandas as pd
```
## Series
A series is the basic one-dimensional data structure in pandas. The constainer consists of an axis label and a numpy array structure.

In general, a pandas series can be created using the corresponding constructor
```Python
class pandas.Series(data=None,		# array-like, Iterable, dict, scalar value
  		    index=None,		# array-like of Index (1-dim)
		    dtype=None,		# str, np.dtype, ExtensionDtype, optional
		    name=None,		# Hashable, default None
	            copy=None,		# bool, default False 
		    fastpath=False)
```
Basically, a series is created via the constructor as follows

A few examples for pandas series
```Python
s1 = pd.Series([1, 3, 5, np.nan, 6, 8])
s2 = pd.Series(np.array([2, 5, 8]), index=["a", "b", "c"])
```
This results in
```
s1
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```
and
```
s2
a    2
b    5
c    8
dtype: int32
```
Additionally, a series can be created from a dictionary
```Python
s3 = pd.Series({"b": 1, "a": 0, "c": 2})
```
### Index and Values
The index and values of a series can be extracted as follows
```Python
pd.Series({"a": 1, "b": 2, "c": 5}).index 	# Index(['a', 'b', 'c'], dtype='object')
pd.Series([1, 2, 3]).index 			# RangeIndex(start=0, stop=3, step=1)

pd.Series({"a": 1, "b": 2, "c": 5}).values	# array([1, 2, 5], dtype=int64)
pd.Series([1, 2, 3]).values			# array([1, 2, 3], dtype=int64)
```
### Value extraction
Values of a series can be extracted as follows
```Python
s4 = pd.Series(np.array([2.0, 5.3]))
s4[1]	# 5.3

s5 = pd.Series({"a": 1, "b": -2.5})
s5[0]	# 1.0
s5["b"] # -2.5
```
#### Examples
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
## Data Frame
A data frame is the basic two-dimensional tabular data structure in pandas. The container consists of an index for rows and columns, respectively.

The constructor of a pandas data frame is given as follows
```Python
class pandas.DataFrame(data=None,	# ndarray (structured or homogeneous), Iterable, dict, or DataFrame
		       index=None,	# Index or array-like
		       columns=None,	# Index or array-like
		       dtype=None,	# dtype, default None
		       copy=None)	# bool or None, default None
```
As an example we consider the following data frame
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
Furthermore, we can create data frames as follows
```Python
d = {"column_1": [1, 2], "column_2": [3, 4]}
df2 = pd.DataFrame(data=d)
```
### Attributes
```Python
DataFrame.index			# The index (row labels) of the DataFrame
DataFrame.columns		# The column labels of the DataFrame.
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
DataFrame.index			# A
```
### Methods

### Add columns
We can add columns via
```Python
df["factor"] = 2.0
df["weight_double"] = df["factor"] * df["weight"]
```
resulting in
```
    name	weight	abs	factor	weight_double
0	A	5	1.25	2.0	10.0
1	B	-2	9.24	2.0	-4.0
2	C	3	-0.12	2.0	6.0
```
We can use the assign and apply methods to create new columns by applying a function to a given column
```Python
df_assigned = df.assign(new_column=df["old_column"].apply(my_function))
df_flag = df.assign(column_flag=df["old_column"].isin(arr_values))
df_where = df.assign(np.where(df["column_valid"], df["value_valid"], "value_not_valid"))
```
We can also include lambda functions as well
```Python
df_lambda = df["new_column"].apply(lambda x: my_function(x) if isinstance(x, str) else "")
```
### Rename
We can rename columns as follows
```Python
df.rename(columns={"factor": "weight_factor"})
```
```
	name	weight	abs	weight_factor	weight_double
0	A	5	1.25	2.0		10.0
1	B	-2	9.24	2.0		-4.0
2	C	3	-0.12	2.0		6.0
```
### Filter
We can filter rows based on conditions as follows
```Python
df_filter = df[df["name"] == "A"].copy()
```
```
df_filter
	name	weight	abs	factor	weight_double
0	A	5	1.25	2.0	10.0
```
### Filling NA values
In order to fill NA values we can use the following procedure
```Python
df["filled_na"] = df["col_with_na"].fillna("to be done")
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
```
df_1 = pd.DataFrame({"id": [1, 2, 3],
                     "value": ["a", "b", "c"]})
df_2 = pd.DataFrame({"id": [2, 3, 4],
                     "value_2": ["d", "e", "f"]})
df_3 = df_1.merge(df_2, how="inner", on="id")
```
This results in
```
df_1
 	id	value_1
0	1	a
1	2	b
2	3	c

df_2
        id	value_2
0	2	d
1	3	e
2	4	f

df_3
	id	value_1	value_2
0	2	b	d
1	3	c	e
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
