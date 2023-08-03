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
A series is the basic one-dimensional data structure in pandas
```Python
s = pd.Series([1, 2, 3])
```
This results in 
```
0    1
1    2
2    3
dtype: int64
```
## Data Frame creation
A data frame is the basic two-dimensional data structure in pandas
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
## Input and Output
```Python
# CSV format
df.load_csv()
df.to_csv()

# Excel format
df.load_excel()
df.to_excel()
```
