**

# Chapter 3: Python Imports Reference

## Table of Contents

- 3.1 Introduction to Python Imports
    
- 3.2 NumPy Imports
    
- 3.3 Pandas Imports
    
- 3.4 Polars Imports
    
- 3.5 DuckDB Imports
    
- 3.6 SciPy Imports
    
- 3.7 Statsmodels Imports
    
- 3.8 Matplotlib Imports
    
- 3.9 Seaborn Imports
    
- 3.10 Plotly Imports
    
- 3.11 Scikit-learn Imports
    
- 3.12 SQLAlchemy Imports
    
- 3.13 PyArrow Imports
    
- 3.14 Utility Imports
    
- 3.15 Import Best Practices
    

---

## 3.1 Introduction to Python Imports

### What is an Import?

An import in Python is a statement that loads code from external modules or packages into your current Python environment. When you import a library, you're essentially telling Python: "Make the functions, classes, and variables defined in this external file available for me to use."

### Why Do Imports Exist?

Python's power comes from its extensive ecosystem of libraries. Instead of writing every function from scratch, you can import battle-tested, optimized code written by experts. This follows the DRY (Don't Repeat Yourself) principle and significantly accelerates development.

### How Imports Work

When you execute an import statement, Python:

1. Searches for the module in:
    

- The current directory
    
- The PYTHONPATH environment variable
    
- The standard library directories
    
- Site-packages (where third-party libraries are installed)
    

3. Compiles the module (if not already compiled)
    
4. Executes the module's code
    
5. Creates a reference to the module in the current namespace
    

### Import Syntax Variations

python

---

#### Basic import - imports the entire module

`import pandas

---
#### Import with alias - common practice for frequently used libraries

`import pandas as pd

---
#### Import specific functions/classes

`from pandas import DataFrame, Series

---
#### Import all functions (generally discouraged)

`from pandas import *

---
#### Import with alias for specific function

`from pandas import DataFrame as DF

---

### Why Different Libraries Use Different Import Aliases

The data science community has established convention aliases that make code more readable:

|   |   |   |
|---|---|---|
|Library|Standard Alias|Reason|
|NumPy|np|Short, memorable|
|Pandas|pd|Historically from "panel data"|
|Matplotlib|plt|From "plot"|
|Seaborn|sns|Named after Samuel Norman Seaborn (a character from The West Wing)|
|Plotly|px|From "plotly express"|
|Scikit-learn|sklearn|Abbreviation of "scikit-learn"|

Interview Note: Interviewers often expect you to use these standard aliases. Using unconventional aliases may suggest you're not familiar with community practices.

### Module vs Package

- Module: A single Python file (.py) containing functions, classes, and variables
    
- Package: A collection of modules organized in directories with an __init__.py file
    

Example: numpy.random is a module (random.py) within the NumPy package.

---

## 3.2 NumPy Imports

### Core Import

python

`import numpy as np

What it imports: The entire NumPy library, which provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.

Why it's used: NumPy is the fundamental package for scientific computing in Python. It provides:

- The ndarray data structure (n-dimensional array)
    
- Vectorized operations (operations applied to entire arrays without explicit loops)
    
- [[Broadcasting]] (performing operations on arrays of different shapes)
    
- Linear algebra, Fourier transform, and random number capabilities
    

When to use: Whenever you need to perform numerical operations on large datasets, especially mathematical computations, statistical analyses, or when you need a fast array data structure.

When to avoid: For data with mixed types (strings, dates, and numbers) - use Pandas instead. For extremely large datasets that don't fit in memory - consider Dask or Vaex.

### Essential Sub-Imports
---

### 3.2.1Random number generation

`from numpy import random as npr

 In Pandas and NumPy, random number generation is essential for creating test data, shuffling records, or picking random samples from your data sheets. 

**Random Number Generation in Data Pipelines -**

To generate random data, you should use **NumPy's modern generator system 
(`np.random.default_rng()`)**, which is the official standard for speed and security.

Core Generation Scenarios

- **`integers(low, high, size)`**: Generates whole numbers within a specific range.
- **`random(size)`**: Generates decimal values between 0.0 and 1.0.
- **`choice(array, size)`**: Picks random items from a custom list.
- **`normal(mean, std_dev, size)`**: Generates data that fits a [[bell curve distribution]]. 


#### 1. Setting a "[[Seed]]" for Reproducibility

When running tests, you want "random" numbers to stay the same every time you run the script so your data pipeline is predictable. You achieve this by passing a fixed number (seed) to the generator.

python

```python
import numpy as np
import pandas as pd

# Initialize the generator with a fixed seed (e.g., 42)
rng = np.random.default_rng(seed=42)
```

Use code with caution.


#### 2. Python Code Examples

**A. Creating a Random Test DataFrame from Scratch**

This snippet dynamically builds a mockup of an Excel sheet with random sales, item choices, and ratings. 

python

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(seed=42)

# Generate raw random sequences
product_pool = ['Laptop', 'Mouse', 'Keyboard', 'Monitor']

mock_data = {
    # 5 random integers between 100 and 1000
    "Transaction_ID": rng.integers(100, 1000, size=5),
    
    # 5 random selections from our list (with replacement)
    "Product": rng.choice(product_pool, size=5),
    
    # 5 random decimal percentages between 0.0 and 1.0
    "Discount_Rate": rng.random(size=5),
    
    # 5 numbers following a standard normal bell curve
    "Performance_Score": rng.normal(loc=0, scale=1, size=5)
}

df = pd.DataFrame(mock_data)
print(df)
```

Output:
```table
   Transaction_ID   Product  Discount_Rate  Performance_Score
0             180   Monitor       0.975622           0.879398
1             796    Laptop       0.761140           0.777792
2             689  Keyboard       0.786064           0.066031
3             494    Laptop       0.128114           1.127241
4             489    Laptop       0.450386           0.467509
```
Use code with caution.

**B. Shuffling or Sampling an Existing Sheet**

If you have a 90-page dataset and want to shuffle the row orders or grab a random slice to print out:

python

```python
# Load your dataset
df = pd.read_excel("large_dataset.xlsx")

# 1. Grab a completely random 10% sample of rows
sample_df = df.sample(frac=0.10, random_state=42)

# 2. Grab exactly 5 random rows
five_rows_df = df.sample(n=5, random_state=42)

# 3. Shuffle all rows randomly (keeping the data intact but changing positions)
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

Use code with caution.

_Note: `random_state` inside Pandas acts as the internal seed configuration tool._


Legacy Warning for Your Notes

Older tutorials might tell you to use `np.random.seed()` or `np.random.randint()`. Avoid using these routines in your code, as they are legacy architectures that run slower and can pollute your global environment state. Always prefer using `np.random.default_rng()`. 

---
---

### 3.2.2 Linear algebra

#### `from numpy import linalg as LA

The command `from numpy import linalg as LA` imports NumPy's **Linear Algebra submodule** and renames it to `LA` for short.

==Linear algebra is the mathematical backbone of machine learning, data science, and advanced data processing==. In data analysis, this module is used to calculate distances between data points vector-by-vector, find patterns in matrices, or solve complex systems of equations.

NumPy Linear Algebra (`numpy.linalg`)

When working with multi-dimensional datasets (matrices), standard loops are too slow. `numpy.linalg` uses optimized C libraries under the hood to perform matrix transformations instantly. 

Core Vector & Matrix Operations

- **`LA.norm(x)`**: Calculates vector length or distance ([[Euclidean distance]]).
- **`LA.inv(x)`**: Calculates the inverse of a matrix (reverses a transformation matrix).
- **`LA.det(x)`**: Finds the determinant of a matrix (scaling factor of a matrix transformation).
- **`LA.eig(x)`**: Calculates Eigenvalues and Eigenvectors (used in [[Principal Component Analysis]] / PCA to reduce dataset sizes).
- **`LA.solve(A, b)`**: Solves linear matrix equations instantly.


Practical Code Examples for Data Notes

#### 1. Calculating Vector Norms (Finding Data Similarities)

The "Norm" is commonly used in recommendation algorithms to find out how similar two rows of data (like user preferences) are to each other.

python

```python
import numpy as np
import numpy.linalg as LA

# Row profiles for User 1 and User 2 based on movie ratings
user1 = np.array([5, 3, 0, 4])
user2 = np.array([4, 3, 1, 4])

# Find the distance between the two vectors
# A smaller norm value means the data profiles are closer together
distance = LA.norm(user1 - user2)
print(f"Euclidean distance between users: {distance:.2f}")
```

Use code with caution.

#### 2. Solving a System of Linear Equations (Business Optmization)

Imagine a business trying to find the individual cost of items based on bulk receipts.

- Receipt 1: 3x + 1y = 9
- Receipt 2: 1x + 2y = 8

Instead of manual algebra, you can pass the coefficients and totals straight to `LA.solve`:

python

```python
import numpy as np
import numpy.linalg as LA

# Coefficients matrix (A)
A = np.array([[3, 1], 
              [1, 2]])

# Totals matrix (b)
b = np.array([9, 8])

# Solve for x and y
solution = LA.solve(A, b)

print(f"Item X cost: {solution[0]}")  # Output: 2.0
print(f"Item Y cost: {solution[1]}")  # Output: 3.0
```

Use code with caution.

Performance Warning for Your Notes

For standard data applications, `numpy.linalg` is perfectly fine. However, if your data pipeline grows to look like a massive neural network or deep learning workflow, look into **`scipy.linalg`**. SciPy includes all of NumPy's features but adds even faster architectures designed for specific matrix shapes. 

---
---

### Masked arrays (handling missing data)

#### `from numpy import ma

The command `from numpy import ma` imports NumPy's specialized **Masked Array module**.

In data processing, you frequently run into datasets with missing, corrupted, or invalid entries (such as negative values for age, empty text cells, or sensor errors). If you pass these broken values into mathematical functions, they will ruin your calculations or cause crashes. 
A **Masked Array** solves this by pairing your raw data array with a second, invisible boolean array called a **Mask**. If a data point is masked (`True`), NumPy completely skips it during mathematical operations without deleting it from your dataset. 


Handling Missing Data with Masked Arrays (`numpy.ma`)

Masked arrays are incredibly useful when you want to preserve the original structure and size of a multi-page data grid, but need to calculate clean metrics (like means or totals) without letting corrupted values skew the results.

Core Architecture

A masked array consists of three layers:

1. **`data`**: The original input array (including the bad values).
2. **`mask`**: A boolean array of identical shape. `True` means **hide/ignore** this value; `False` means **keep** it.
3. **`fill_value`**: A default fallback value used if you decide to convert the masked array back into a standard array later. 

Python Code Examples

#### 1. Masking Invalid Values Automatically

This snippet shows how to mask corrupted data points (like sensor errors flagged as negative numbers) so they do not ruin an average.

python

```python
import numpy as np
from numpy import ma

# Raw sensor data where -999.0 represents a system error/missing data
raw_readings = np.array([22.5, 23.1, -999.0, 24.0, -999.0, 21.8])

# 1. Create a masked array where any value equal to -999.0 is hidden
masked_readings = ma.masked_equal(raw_readings, -999.0)

print("Masked Array Visualized:")
print(masked_readings) 
# Output: [22.5 23.1 -- 24.0 -- 21.8]  (-- indicates a hidden value)

# 2. Calculate the mean
# Standard np.mean() would include -999.0 and ruin the math.
# ma.mean() automatically skips the hidden values.
print("\nClean Average Temperature:", masked_readings.mean())
# Output: 22.85
```

Use code with caution.

#### 2. Custom Masking Using Conditions

You can build custom masks using logic statements (like masking values outside a specific acceptable operating boundary). 

python

```python
import numpy as np
from numpy import ma

# Raw data sheet
data = np.array([10, 50, 200, 30, 400, 15])

# Mask any value that is greater than 100 (out-of-bounds anomalies)
masked_data = ma.masked_where(data > 100, data)

print("Filtered Array:", masked_data)
# Output: [10 50 -- 30 -- 15]

# 3. Exporting data out by filling the hidden slots
# Replaces all hidden '--' values with a clean default value (like 0)
clean_standard_array = masked_data.filled(0)
print("Filled Export Array:", clean_standard_array)
# Output: [10, 50, 0, 30, 0, 15]
```

Use code with caution.


When to Use It 🚀

- **Preserving Array Shapes:** When working with image grids or matrix layouts where deleting a missing row would break the shape of your matrix. 
- **Working with Missing Sensor Flags:** When hardware devices use specific boundary numbers (like `-999`, `999`, or `0`) to communicate a data failure. 
- **Fast Conditional Math:** When you want to run quick math on subset ranges without wasting processing power creating brand new sliced copies of your data.


When NOT to Use It ⚠️

- **When Working in Pandas:** If your 90-page dataset is already loaded inside a Pandas DataFrame, do not use `numpy.ma`. Pandas has its own native, highly optimized missing data engine using **`NaN`** (Not a Number) values and built-in cleanup functions like `df.dropna()` and `df.fillna()`. 
- **For Permanent Data Pruning:** If you never intend to use the corrupted rows again, it is more memory-efficient to drop them completely using array slicing rather than storing a permanent mask layer in your computer's RAM.

#### NumPy Built-in Masking Functions Reference

These functions allow you to automatically generate masks based on specific conditions, data errors, or numeric ranges. 

|Function Name|What it Masks / Hides|Best Use Case|
|---|---|---|
|**`ma.masked_invalid(arr)`**|Hides all `NaN` (Not a Number) and `Inf` (Infinity) entries.|Cleaning up broken mathematical outputs or corrupted file imports.|
|**`ma.masked_equal(arr, value)`**|Hides data points that exactly match a specific target value.|Hiding hardcoded hardware error codes like `-999` or `0`.|
|**`ma.masked_where(condition, arr)`**|Hides any data point that evaluates to `True` based on a custom logic statement.|Hiding complex logic blocks (e.g., `arr < 0` or `arr > 100`).|
|**`ma.masked_inside(arr, v1, v2)`**|Hides all values that fall **inside** an inclusive range `[v1, v2]`.|Filtering out standard baseline activity to isolate extreme spikes.|
|**`ma.masked_outside(arr, v1, v2)`**|Hides all values that fall **outside** an inclusive range `[v1, v2]`.|Stripping away data anomalies that breach your strict safety limits.|
|**`ma.masked_greater(arr, value)`**|Hides values strictly greater than the threshold (`> value`).|Capping data ceilings (e.g., removing impossible speed readings).|
|**`ma.masked_less(arr, value)`**|Hides values strictly less than the threshold (`< value`).|Dropping floor anomalies (e.g., removing negative prices or ages).|


Quick-Syntax Cheat Sheet

python

```python
import numpy as np
import numpy.ma as ma

# Raw data matrix with a mix of invalid data types, errors, and values
raw_data = np.array([15.0, np.nan, -99.0, 42.0, np.inf, 12.0])

# 1. Clean up invalid math markers (NaN and Inf disappear)
clean_invalid = ma.masked_invalid(raw_data)
# Output: [15.0 -- -99.0 42.0 -- 12.0]

# 2. Hide values inside a specific range (e.g., values between 10 and 20)
clean_range = ma.masked_inside(raw_data, 10, 20)
# Output: [-- np.nan -99.0 42.0 np.inf --]
```

---  
---

### Polynomial operations

#### `from numpy import polynomial as poly


## Frequently Used Functions and Their Imports

## Array Creation Functions

python

``import numpy as np
### Create arrays with specific values

np.array([1, 2, 3])                    # From list

np.zeros((3, 4))                        # Array of zeros

np.ones((2, 3))                         # Array of ones

np.empty((3, 3))                        # Uninitialized array (fast)

np.full((2, 2), 7)                      # Array filled with 7

np.eye(4)                               # Identity matrix

np.identity(3)                          # Identity matrix (alternative)


* **`np.array([1, 2, 3])`**: Converts a standard Python list into a 1D NumPy data array.
* **`np.zeros((3, 4))`**: Creates a grid of 3 rows and 4 columns entirely filled with the number 0.
* **`np.ones((2, 3))`**: Creates a grid of 2 rows and 3 columns entirely filled with the number 1.
* **`np.empty((3, 3))`**: Allocates memory for a 3x3 grid instantly without clearing it, meaning it contains random placeholder data.
* **`np.full((2, 2), 7)`**: Creates a 2x2 grid where every single cell is filled with the custom number 7.
* **`np.eye(4)`**: Creates a 4x4 square grid with a diagonal line of 1s from top-left to bottom-right and 0s everywhere else.
* **`np.identity(3)`**: Creates a 3x3 square identity matrix with a diagonal of 1s, working identically to np.eye.


### Range-based arrays

np.arange(10)                           # 0 to 9

np.arange(2, 10, 2)                     # 2 to 8 step 2

np.linspace(0, 10, 5)                   # 5 evenly spaced points

np.logspace(0, 3, 4)                    # Logarithmically spaced


* **`np.arange(10)`**: Generates a 1D array of integers starting from 0 up to, but not including, 10.
* **`np.arange(2, 10, 2)`**: Generates integers starting at 2 up to 10, skipping numbers by a step size of 2.
* **`np.linspace(0, 10, 5)`**: Creates an array of exactly 5 numbers evenly spaced between the ranges of 0 and 10 inclusive.
* **`np.logspace(0, 3, 4)`**: Creates an array of 4 points spaced evenly on a logarithmic scale from 10^0 (1) to 10^3 (1000).


### Random arrays

np.random.rand(3, 4)                    # Uniform [0,1)

np.random.randn(3, 4)                   # Standard normal

np.random.randint(0, 10, (3, 4))        # Random integers

np.random.normal(0, 1, (3, 4))          # Normal distribution

np.random.uniform(0, 1, (3, 4))         # Uniform distribution

## Array Manipulation Functions

python

### Reshaping

arr.reshape(2, 6)                       # Change shape

arr.flatten()                           # Flatten to 1D

arr.ravel()                             # Flatten (returns view when possible)

arr.transpose()                         # Transpose matrix

arr.T                                   # Transpose (shortcut)

arr.swapaxes(0, 1)                      # Swap axes

  

### Combining arrays

np.concatenate((arr1, arr2), axis=0)    # Join arrays

np.vstack((arr1, arr2))                 # Stack vertically

np.hstack((arr1, arr2))                 # Stack horizontally

np.column_stack((arr1, arr2))           # Stack as columns

np.row_stack((arr1, arr2))              # Stack as rows

  

### Splitting arrays

np.split(arr, 3)                        # Split into 3 parts

np.hsplit(arr, 3)                       # Split horizontally

np.vsplit(arr, 3)                       # Split vertically

## Mathematical Functions

python

### Basic operations

np.add(arr1, arr2)                      # Element-wise addition

np.subtract(arr1, arr2)                 # Element-wise subtraction

np.multiply(arr1, arr2)                 # Element-wise multiplication

np.divide(arr1, arr2)                   # Element-wise division

np.power(arr, 2)                        # Element-wise power

np.sqrt(arr)                            # Square root

np.exp(arr)                             # Exponential

np.log(arr)                             # Natural log

np.log10(arr)                           # Log base 10

np.log2(arr)                            # Log base 2

  

## Trigonometric

np.sin(arr)                             # Sine

np.cos(arr)                             # Cosine

np.tan(arr)                             # Tangent

np.arcsin(arr)                          # Arc sine

np.arccos(arr)                          # Arc cosine

np.arctan(arr)                          # Arc tangent

  

## Statistical

np.mean(arr)                            # Mean

np.median(arr)                          # Median

np.std(arr)                             # Standard deviation

np.var(arr)                             # Variance

np.min(arr)                             # Minimum

np.max(arr)                             # Maximum

np.percentile(arr, 50)                  # Percentile

np.quantile(arr, 0.5)                   # Quantile

np.corrcoef(arr1, arr2)                 # Correlation coefficient

np.cov(arr1, arr2)                      # Covariance

  

## Aggregation

np.sum(arr)                             # Sum

np.prod(arr)                            # Product

np.cumsum(arr)                          # Cumulative sum

np.cumprod(arr)                         # Cumulative product

np.any(arr > 0)                         # Any True

np.all(arr > 0)                         # All True

## Linear Algebra Functions

python

### Dot product and matrix multiplication

np.dot(arr1, arr2)                      # Dot product

arr1 @ arr2                             # Matrix multiplication (Python 3.5+)

  

### Matrix operations

np.linalg.inv(arr)                      # Matrix inverse

np.linalg.det(arr)                      # Determinant

np.linalg.eig(arr)                      # Eigenvalues and eigenvectors

np.linalg.eigvals(arr)                  # Eigenvalues only

np.linalg.svd(arr)                      # Singular value decomposition

np.linalg.solve(A, b)                   # Solve linear system Ax = b

np.linalg.lstsq(A, b)                   # Least squares solution

np.linalg.norm(arr)                     # Matrix norm

np.linalg.qr(arr)                       # QR decomposition

np.linalg.cholesky(arr)                 # Cholesky decomposition

np.linalg.matrix_power(arr, 3)          # Matrix power

## Broadcasting and Vectorization

python

### Broadcasting - operations on arrays of different shapes

arr = np.array([1, 2, 3])

arr + 10                                # [11, 12, 13]

arr * 2                                 # [2, 4, 6]

arr.reshape(3, 1) + np.array([10, 20])  # Broadcast to 3x2

  

# Vectorization - operations applied to entire arrays

# Instead of:

for i in range(len(arr)):

    arr[i] = arr[i] ** 2

  

# Use:

arr ** 2                                # Much faster

### Common Mistakes with NumPy Imports

1. Creating Python lists instead of NumPy arrays

python

# Wrong

arr = [1, 2, 3]                         # This is a Python list

  

# Correct

arr = np.array([1, 2, 3])               # This is a NumPy array

2. Confusing array copy vs view

python

arr = np.array([1, 2, 3, 4, 5])

# This creates a view (changes affect original)

view = arr[1:4]

view[0] = 99                             # Changes arr[1] to 99

  

# This creates a copy (changes don't affect original)

copy = arr[1:4].copy()

copy[0] = 99                             # arr[1] remains unchanged

3. Forgetting that arrays use element-wise operations

python

a = np.array([1, 2])

b = np.array([3, 4])

# Element-wise multiplication

result = a * b                           # [3, 8]

  

# For matrix multiplication, use @

result = a @ b                           # Error (shapes don't match)

Best Practice: Always use np.array() for data you want to perform numerical operations on, and use Python lists only for simple, non-numerical data structures.

---

## 3.3 Pandas Imports

### Core Import

python

import pandas as pd

What it imports: The entire Pandas library, which provides high-performance, easy-to-use data structures and data analysis tools for Python.

Why it's used: 
Pandas is the Swiss Army knife of data analysis. It provides:

- DataFrame: A 2D labeled data structure with columns of potentially different types
    
- Series: A 1D labeled array
    
- Data alignment and handling of missing data
    
- Powerful data manipulation (grouping, merging, reshaping)
    
- Time series functionality
    
- Input/output tools for reading and writing data
    

When to use: 
Almost always for data analysis tasks. Pandas is ideal for:

- Data cleaning and preparation
    
- Exploratory data analysis
    
- Handling tabular data with mixed types
    
- Working with time series data
    
- Data aggregation and transformation
    

When to avoid:

- When performance and memory usage are critical (use Polars)
    
- For very large datasets that don't fit in memory (use Dask or DuckDB)
    
- When you only need array operations (use NumPy)
    
- When you need to work with SQL databases directly (use SQLAlchemy)
    

### Essential Sub-Imports

python

### 1. IO tools for reading various file formats

- from `pandas.io import ExcelFile, ExcelWriter`

---

Working with Excel Files in Python (Pandas & ExcelWriter)

Core Libraries Needed

To read and write Excel files, you need `pandas` along with engine libraries like `openpyxl` (for `.xlsx` files).

bash

```
pip install pandas openpyxl
```

Use code with caution.


1. Importing / Reading Excel Files

You use `pd.read_excel()` to load data into a DataFrame. 

Basic Import

python

```
import pandas as pd

# Load the first sheet of an Excel file
df = pd.read_excel("data_file.xlsx")
```

Use code with caution.

Advanced Import Options

python

```
# Read a specific sheet by its name or index position
df = pd.read_excel("data_file.xlsx", sheet_name="Sales_Data")  # By name
df = pd.read_excel("data_file.xlsx", sheet_name=0)             # First sheet (0-indexed)

# Read multiple sheets at once (returns a dictionary of DataFrames)
all_sheets = pd.read_excel("data_file.xlsx", sheet_name=["Jan", "Feb"])
jan_df = all_sheets["Jan"]

# Skip rows or select specific columns
df = pd.read_excel("data_file.xlsx", skiprows=2, usecols="A:C")
```

Use code with caution.


2. Exporting to Excel using ExcelWriter

If you only need to output **one DataFrame to one sheet**, use the basic `to_excel()` function:

python

```
df.to_excel("output.xlsx", sheet_name="Sheet1", index=False)
```

Use code with caution.

Why use `pd.ExcelWriter`?

You must use `pd.ExcelWriter` when you want to perform advanced operations, such as:

1. Writing **multiple DataFrames** into different sheets within the **same workbook**.
2. **Appending** a new sheet to an existing Excel file without overwriting old data. ]

Method A: Writing Multiple Sheets (New File)

Using a `with` statement ensures the file saves and closes automatically without corruption. 

python

```
import pandas as pd

# Sample DataFrames
df_sales = pd.DataFrame({"Item": ["A", "B"], "Amount": [100, 150]})
df_costs = pd.DataFrame({"Item": ["A", "B"], "Cost": [40, 60]})

# Create the writer object
with pd.ExcelWriter("company_report.xlsx", engine="openpyxl") as writer:
    df_sales.to_excel(writer, sheet_name="Sales", index=False)
    df_costs.to_excel(writer, sheet_name="Costs", index=False)
```

Use code with caution.

Method B: Appending to an Existing Excel File

To add a sheet to a file that already exists on your computer without deleting its current contents, use `mode="a"`. 

python

```
import pandas as pd

df_new_data = pd.DataFrame({"Item": ["C"], "Amount": [200]})

# Open the file in append mode
with pd.ExcelWriter("company_report.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_new_data.to_excel(writer, sheet_name="March_Sales", index=False)
```

Use code with caution.

_Note on `if_sheet_exists`: Options include `"replace"` (overwrites the sheet if the name matches) or `"new"` (creates a unique name like Sheet_1)._ 

---

- from `pandas import read_csv, read_excel, read_sql`

  ---
  

### 2. Plotting functionality (uses Matplotlib)

`import pandas.plotting as pdp

---
The command `import pandas.plotting as pdp` imports a specific **advanced visualization submodule** from the Pandas library and renames it to `pdp` for short.

While you can usually create basic graphs in Pandas using just `df.plot()`, this specific submodule is used for **complex statistical plotting**.

What is inside `pandas.plotting`?

It contains special tools designed for multi-variable data analysis. If you paste this into your Obsidian notes, the most common functions it provides are:

- **`pdp.scatter_matrix(df)`**: Creates a grid of scatter plots comparing every numerical column against every other column. Perfect for spotting hidden relationships in your Excel data.
- **`pdp.parallel_coordinates(df, 'class_column')`**: Excellent for comparing multiple columns at once across different categories.
- **`pdp.radviz(df, 'class_column')`**: Plots multi-dimensional data onto a 2D circle format.
- **`pdp.bootstrap_plot(df['column'])`**: Used by statisticians to estimate uncertainty by repeatedly sampling your data. 

Simple Example for Your Notes

python

```
import pandas as pd
import pandas.plotting as pdp
import matplotlib.pyplot as plt

# 1. Load data from your Excel file
df = pd.read_excel("sales_data.xlsx")

# 2. Generate a grid of plots checking correlations across all data
pdp.scatter_matrix(df, figsize=(6, 6))

# 3. Display the graph
plt.show()
```

Use code with caution.

Why use `as pdp`?

It is simply a shortcut alias. Instead of typing the long `pandas.plotting.scatter_matrix()`, you can just type `pdp.scatter_matrix()`, keeping your Python code clean and easy to read. 

If you want, let me know if you need:

- The syntax for a **specific type of chart** (like a histogram grid or trendline)
- To know how to **save these generated graphs directly back into your Excel file sheets**

---

---

### 3. Saving Advanced Pandas Charts Directly into Excel Sheets

Core Requirements

You need `matplotlib` to handle saving the image object, and `openpyxl` as the layout engine. 

bash

```
pip install pandas openpyxl matplotlib
```

Use code with caution.

The Code Solution

This script creates statistical plots using `pandas.plotting`, exports a raw data sheet, and appends the actual graphical figure into a designated cell location. 
python

```
import pandas as pd
import pandas.plotting as pdp
import matplotlib.pyplot as plt
import openpyxl

# 1. Prepare sample data
data = {
    "Sales": [120, 150, 90, 210, 180, 250],
    "Marketing_Spend": [15, 20, 12, 35, 25, 40],
    "Profit": [30, 45, 15, 70, 50, 85]
}
df = pd.DataFrame(data)

# 2. Generate an advanced multi-variable scatter matrix
pdp.scatter_matrix(df, figsize=(8, 8), diagonal='kde', color='purple')

# 3. Save the active plot to your computer as a static image
temp_image_path = "scatter_matrix_plot.png"
plt.tight_layout()
plt.savefig(temp_image_path, dpi=150)
plt.close() # Free memory layer

# 4. Open ExcelWriter using the openpyxl engine
excel_file = "business_report.xlsx"
with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    # Write raw data block to sheet 1
    df.to_excel(writer, sheet_name="Data", index=False)
    
    # Access the raw openpyxl worksheet structure directly
    workbook = writer.book
    worksheet = writer.sheets["Data"]
    
    # Load and anchor the image into a specific cell block
    img = openpyxl.drawing.image.Image(temp_image_path)
    img.anchor = "F2"  # Image sits 5 columns right of the raw data
    
    # Inject image into sheet
    worksheet.add_image(img)
```

Use code with caution.

Troubleshooting Layout Controls

1. The Overlap Warning

By default, the image top-left pixel anchors directly inside your chosen cell coordinate. Ensure `img.anchor` is placed safely away from columns filled by `df.to_excel` to prevent the picture from pasting over text blocks. []

2. Adjusting Image Sizing

If the generated graph layout feels too oversized or squished inside your Excel layout grid: 

- Adjust the graph dimensions globally before export using the `figsize=(width, height)` variable within your mapping code block.
- Control pixels directly by adding spacing metrics:

python

```
img.width = 600  # Set custom pixel width
img.height = 600 # Set custom pixel height
```

Use code with caution.

---
---

### Options for display settings

`from pandas import set_option, reset_option, get_option

  This command imports specific configuration tools from Pandas that allow you to change how data is displayed on your screen.

When working with large data files (like your 90-page document), Pandas automatically hides rows and columns using ellipses (`...`) to keep your screen tidy. By using `set_option`, you can force Pandas to show every single row, column, or long text block without cutting anything off. 
Here is a clean, structured summary formatted perfectly for your Obsidian vault:


Controlling Pandas Display Options

These functions let you view massive datasets or long code blocks in your terminal or Jupyter notebooks without data truncation.

The Core Commands

- **`set_option(option_name, value)`**: Temporarily updates a specific setting.
- **`get_option(option_name)`**: Checks the current value of a configuration setting.
- **`reset_option(option_name)`**: Restores a specific setting back to its original default behavior. 


Essential Settings for Large Data Pipelines

You can pass these specific text flags into your configuration commands:

|Option Flag|What it Controls|Default Value|Recommended Value for Audits|
|---|---|---|---|
|`'display.max_rows'`|Max rows printed before hiding data|`60`|`None` (Shows everything)|
|`'display.max_columns'`|Max columns printed before hiding data|`20`|`None` (Shows everything)|
|`'display.max_colwidth'`|Character limit before clipping long text strings|`50`|`None` (Shows full sentences)|
|`'display.precision'`|Number of decimal places displayed for floating numbers|`6`|`2` (Great for currency calculations)|


Python Code Template

python

```
import pandas as pd
from pandas import set_option, get_option, reset_option

# 1. Check current limits
print("Default Max Rows:", get_option('display.max_rows'))

# 2. Force Pandas to print EVERYTHING without clipping data
set_option('display.max_rows', None)       # Unlocks unlimited rows
set_option('display.max_columns', None)    # Unlocks unlimited columns
set_option('display.max_colwidth', None)   # Unlocks full text strings
set_option('display.precision', 2)         # Formats floats to 2 decimal places

# Sample large dataframe that would normally be clipped with "..."
df = pd.read_excel("large_company_dump.xlsx")
print(df) # Everything prints fully to the console

# 3. Reset a specific configuration back to defaults
reset_option('display.max_rows')

# 4. Reset ALL configuration settings back to defaults at once
reset_option('all')
```

Use code with caution.


Pro-Tip: Clean Context Isolation

If you only want to change display behaviors for a single snippet without permanently modifying your global script session, run your code inside an isolated **context manager block**:

python

```
# The configurations inside this block expire automatically when the block finishes
with pd.option_context('display.max_rows', 10, 'display.max_columns', 5):
    df_preview = pd.read_excel("data.xlsx")
    print(df_preview) 
```

Use code with caution.

---
----

### Testing utilities

`from pandas import testing as pdt

  The command `from pandas import testing as pdt` imports Pandas' built-in **testing suite module** and renames it to `pdt`.

This module is not used for data cleaning or analysis. Instead, it is a critical tool for **Unit Testing**. It allows you to programmatically verify that two DataFrames, Series, or Indexes are **exactly identical** in data values, data types, and index orders. 

Here is a clean, structured breakdown you can copy straight into your Obsidian notes:


Unit Testing in Pandas (`pandas.testing`)

When writing data pipelines, you need to ensure your functions do not accidentally alter your data structures. Using standard comparison tools like `df1 == df2` returns a matrix of True/False values, which triggers errors in testing frameworks. `pandas.testing` provides functions that assert equality directly. 
Core Assert Functions

- **`pdt.assert_frame_equal(df1, df2)`**: Compares two DataFrames.
- **`pdt.assert_series_equal(s1, s2)`**: Compares two Pandas Series.
- **`pdt.assert_index_equal(idx1, idx2)`**: Compares two Index arrays.


Code Example for Your Notes

This shows how to write a test case to verify that an Excel cleanup function works correctly without altering data types or column names.

python

```
import pandas as pd
import pandas.testing as pdt

# 1. Define the actual function you want to test
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

# 2. Set up the input testing data
dirty_df = pd.DataFrame({"  Sales  ": , " Profit ": })

# 3. Create the manually verified "Expected Output" matrix
expected_df = pd.DataFrame({"sales": , "profit": })

# 4. Run the function to get the "Actual Output"
actual_df = clean_column_names(dirty_df)

# 5. Assert equality using pdt
pdt.assert_frame_equal(actual_df, expected_df)
print("Test Passed: The DataFrames are perfectly identical!")
```

Use code with caution.


Critical Parameters for Loose Testing

By default, `assert_frame_equal` is incredibly strict. It will fail the test if one column is float64 and the other is int64, even if the numbers match. You can loosen the rules using these parameters:

python

```
# Check values but ignore if one is float and the other is integer
pdt.assert_frame_equal(df1, df2, check_dtype=False)

# Ignore the exact sorting order of the columns
pdt.assert_frame_equal(df1, df2, check_like=True)

# Allow minor decimal rounding differences (e.g., currency rounding)
pdt.assert_frame_equal(df1, df2, atol=0.01)
```

Use code with caution.

---

### Date offset functionality

`from pandas.tseries import offsets

### Frequently Used Functions and Their Imports

#### Reading Data

python

`import pandas as pd

  

# Reading data from files

`df = pd.read_csv('data.csv')                     # CSV file

`df = pd.read_excel('data.xlsx', sheet_name=0)    # Excel file

`df = pd.read_json('data.json')                   # JSON file

`df = pd.read_html('data.html')[0]                # HTML tables

`df = pd.read_sql('SELECT * FROM table', conn)    # SQL database

`df = pd.read_pickle('data.pkl')                  # Python pickle

`df = pd.read_parquet('data.parquet')             # Parquet file

`df = pd.read_feather('data.feather')             # Feather file

  

# Reading with options

`df = pd.read_csv('data.csv',

                 sep=',',                        # Separator

                 header=0,                       # Row to use as column names

                 index_col=0,                    # Column to use as index

                 skiprows=[1, 2],                # Rows to skip

                 nrows=1000,                     # Number of rows to read

                 dtype={'column': 'float64'},    # Column data types

                 parse_dates=['date_column'],    # Parse dates

                 encoding='utf-8',               # Character encoding

                 na_values=['NA', 'null'])       # Values to treat as NaN

#### Creating DataFrames and Series

python

# Creating from dictionaries

df = pd.DataFrame({

    'name': ['Alice', 'Bob', 'Charlie'],

    'age': [25, 30, 35],

    'salary': [50000, 60000, 70000]

})

  

# Creating from lists

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])

  

# Creating from NumPy arrays

import numpy as np

df = pd.DataFrame(np.random.randn(3, 4), columns=['A', 'B', 'C', 'D'])

  

# Creating Series

s = pd.Series([1, 3, 5, np.nan, 6, 8])

s = pd.Series({'a': 1, 'b': 2, 'c': 3})

  

# Creating with custom index

df = pd.DataFrame(data, index=['row1', 'row2', 'row3'])

#### Data Inspection Functions

python

# Basic inspection

df.head()                           # First 5 rows

df.head(10)                         # First 10 rows

df.tail()                           # Last 5 rows

df.tail(3)                          # Last 3 rows

df.info()                           # Data types and memory usage

df.describe()                       # Statistical summary

df.describe(include='all')          # Includes categorical columns

  

# Shape and dimensions

df.shape                            # (rows, columns)

df.size                             # Number of elements

df.ndim                             # Number of dimensions

  

# Column and index information

df.columns                          # Column names

df.index                            # Row index

df.dtypes                           # Data types of columns

df.axes                             # Axes of DataFrame

  

# Value counts (for categorical columns)

df['column'].value_counts()         # Frequency of unique values

df['column'].value_counts(normalize=True)  # Proportions

  

# Null values

df.isnull()                         # Boolean mask of null values

df.isnull().sum()                   # Count null values per column

df.notnull()                        # Boolean mask of non-null values

#### Data Manipulation Functions

python

# Selecting columns

df['column']                       # Single column as Series

df[['col1', 'col2']]               # Multiple columns as DataFrame

df.column                          # Single column (if name is valid Python variable)

  

# Selecting rows (by index)

df.loc['row_label']                # By label

df.loc[0:5]                        # By label range (inclusive)

df.iloc[0]                         # By integer position

df.iloc[0:5]                       # By integer position range (exclusive)

df.iloc[:, 0:3]                    # Specific rows and columns

  

# Conditional selection

df[df['age'] > 25]                 # Rows where age > 25

df[(df['age'] > 25) & (df['salary'] > 50000)]  # Multiple conditions

  

# Query method (more readable)

df.query('age > 25 and salary > 50000')

df.query('age in [25, 30, 35]')

  

# Adding/Modifying columns

df['new_column'] = df['col1'] + df['col2']

df['new_column'] = df['col1'].apply(lambda x: x**2)

df['new_column'] = np.where(df['col1'] > 0, 'positive', 'negative')

  

# Removing columns/rows

df.drop('column', axis=1)          # Drop column

df.drop(['col1', 'col2'], axis=1)  # Drop multiple columns

df.drop(0, axis=0)                 # Drop row by index

df.drop([0, 1, 2], axis=0)         # Drop multiple rows

#### Data Cleaning Functions

python

# Handling null values

df.dropna()                        # Drop rows with any null

df.dropna(subset=['col1', 'col2']) # Drop rows with null in specific columns

df.dropna(axis=1)                  # Drop columns with any null

df.fillna(0)                       # Fill null with 0

df.fillna(method='ffill')          # Forward fill

df.fillna(method='bfill')          # Backward fill

df.interpolate()                   # Interpolate missing values

  

# Handling duplicates

df.duplicated()                    # Boolean mask of duplicates

df.duplicated(subset=['col1'])     # Check duplicates in specific columns

df.drop_duplicates()               # Remove duplicate rows

df.drop_duplicates(subset=['col1']) # Remove duplicates in specific columns

  

# Data type conversion

df['column'].astype('float64')     # Convert to float

df['column'].astype('int64')       # Convert to int

df['column'].astype('str')         # Convert to string

pd.to_datetime(df['date'])         # Convert to datetime

pd.to_numeric(df['col'], errors='coerce')  # Convert to numeric

  

# String operations

df['text'].str.lower()             # Convert to lowercase

df['text'].str.upper()             # Convert to uppercase

df['text'].str.strip()             # Remove whitespace

df['text'].str.contains('pattern') # Check if contains pattern

df['text'].str.extract('(\\d+)')   # Extract pattern

df['text'].str.split(',')          # Split string

#### Data Aggregation Functions

python

# Basic aggregation

df.mean()                           # Mean of all numeric columns

df['col'].sum()                     # Sum of column

df['col'].min()                     # Minimum

df['col'].max()                     # Maximum

df['col'].median()                  # Median

df['col'].std()                     # Standard deviation

df['col'].var()                     # Variance

df['col'].count()                   # Count non-null values

df['col'].unique()                  # Unique values

df['col'].nunique()                 # Number of unique values

  

# Groupby operations

df.groupby('category')['value'].mean()  # Mean by category

df.groupby('category')[['col1', 'col2']].agg(['mean', 'std'])  # Multiple aggregations

df.groupby('category').agg({

    'col1': 'mean',

    'col2': ['sum', 'std']

})  # Different aggregations for different columns

  

# Pivot tables

df.pivot_table(values='value', 

               index='category', 

               columns='year', 

               aggfunc='mean')

df.pivot_table(values='value', 

               index='category', 

               aggfunc=['mean', 'sum'])

  

# Cross tabulation

pd.crosstab(df['category'], df['region'])

pd.crosstab(df['category'], df['region'], normalize='index')

#### Merging and Joining Functions

python

# Merging DataFrames

pd.merge(df1, df2, on='key')                # Inner join on key

pd.merge(df1, df2, on='key', how='left')    # Left join

pd.merge(df1, df2, on='key', how='right')   # Right join

pd.merge(df1, df2, on='key', how='outer')   # Outer join

pd.merge(df1, df2, left_on='key1', right_on='key2')  # Different key columns

  

# Concatenating DataFrames

pd.concat([df1, df2])                       # Row-wise concatenation

pd.concat([df1, df2], axis=1)               # Column-wise concatenation

pd.concat([df1, df2], ignore_index=True)    # Reset index

  

# Joining DataFrames

df1.join(df2, on='key')                     # Join using index or key

df1.join(df2, how='left')                   # Left join

#### Time Series Functions

python

# Date and time operations

pd.to_datetime('2024-01-01')                # Convert to datetime

pd.date_range('2024-01-01', periods=100)    # Create date range

pd.date_range('2024-01-01', '2024-12-31', freq='M')  # Monthly frequency

  

# Resampling

df.resample('D').mean()                     # Resample to daily frequency

df.resample('M').sum()                      # Resample to monthly

df.resample('Q').std()                      # Resample to quarterly

  

# Shifting and lagging

df['col'].shift(1)                          # Lag by 1 period

df['col'].shift(-1)                         # Lead by 1 period

df['col'].diff()                            # Difference between periods

df['col'].pct_change()                      # Percentage change

  

# Rolling operations

df['col'].rolling(window=7).mean()          # 7-day rolling average

df['col'].rolling(window=30).std()          # 30-day rolling standard deviation

df['col'].expanding().mean()                # Expanding mean

### Common Mistakes with Pandas Imports

1. Forgetting to import specific modules

python

# Wrong - trying to use ExcelWriter without importing

import pandas as pd

with pd.ExcelWriter('file.xlsx') as writer:    # This will work since ExcelWriter is in pd

    df.to_excel(writer)

  

# Correct - but you need to import the submodule

from pandas.io import ExcelFile

2. Using .copy() to avoid SettingWithCopyWarning

python

# This can cause SettingWithCopyWarning

df_subset = df[df['age'] > 25]

df_subset['new_col'] = 0                      # Warning!

  

# Correct - use .copy()

df_subset = df[df['age'] > 25].copy()

df_subset['new_col'] = 0                      # No warning

3. Confusing loc with iloc

python

# Wrong - if index is not integer

df.loc[0]  # Error if index is string

  

# Correct - use iloc for integer positions

df.iloc[0]  # Always gives first row

  

# Correct - use loc for labels

df.loc['row_label']  # Gets row with label 'row_label'

Interview Note: Interviewers often ask about the difference between .loc, .iloc, and direct indexing, as well as when to use each.

Best Practice: Always use pd.read_csv() for CSV files, but consider pd.read_parquet() or pd.read_feather() for larger datasets as they are faster and more memory-efficient.

---

## 3.4 Polars Imports

### Core Import

python

import polars as pl

What it imports: The entire Polars library, which provides a fast DataFrame library implemented in Rust with a Python interface.

Why it's used: Polars was created to address Pandas' performance limitations, especially with large datasets. It offers:

- Blazing speed: Uses all CPU cores and vectorized operations
    
- Memory efficiency: Uses Arrow's memory model
    
- Lazy evaluation: Optimizes queries before execution
    
- Expressive query language: Chainable methods
    
- Better handling of larger-than-memory datasets
    

When to use:

- When performance is critical
    
- Working with datasets larger than 1 GB
    
- Complex data transformations that can be optimized
    
- Building data pipelines where speed matters
    

When to avoid:

- When compatibility with existing Pandas code is needed
    
- When using libraries that only work with Pandas DataFrames
    
- When you need the extensive ecosystem of Pandas
    
- For exploratory analysis where speed isn't critical
    

### Essential Sub-Imports

python

# For working with SQL

import polars.sql as plsql

  

# For string operations

import polars.string as plstr

  

# For working with strings (alternative)

import polars as pl

from polars import col, lit  # For column and literal references

### Frequently Used Functions and Their Imports

#### Reading Data

python

import polars as pl

  

# Reading data from files

df = pl.read_csv('data.csv')                    # CSV file

df = pl.read_excel('data.xlsx')                 # Excel file (requires xlsx2csv)

df = pl.read_parquet('data.parquet')            # Parquet file

df = pl.read_json('data.json')                  # JSON file

df = pl.read_ipc('data.feather')                # Feather/IPC file

df = pl.read_sql('SELECT * FROM table', conn)   # SQL database

df = pl.read_ndjson('data.ndjson')              # Newline-delimited JSON

  

# Reading with options

df = pl.read_csv('data.csv',

                 sep=',',                        # Separator

                 has_header=True,                # Has header row

                 infer_schema_length=1000,       # Rows to infer schema

                 null_values=['NA', 'null'],     # Values to treat as null

                 ignore_errors=True)             # Skip invalid rows

#### Creating DataFrames

python

# Creating from dictionaries

df = pl.DataFrame({

    'name': ['Alice', 'Bob', 'Charlie'],

    'age': [25, 30, 35],

    'salary': [50000, 60000, 70000]

})

  

# Creating from lists

df = pl.DataFrame([

    pl.Series('A', [1, 2, 3]),

    pl.Series('B', [4, 5, 6])

])

  

# Creating from NumPy arrays

import numpy as np

df = pl.DataFrame({

    'A': np.array([1, 2, 3]),

    'B': np.array([4, 5, 6])

})

  

# Creating Series

s = pl.Series('name', ['Alice', 'Bob', 'Charlie'])

s = pl.Series([1, 3, 5, None, 6, 8])  # Auto-named

#### Data Inspection Functions

python

# Basic inspection

df.head()                           # First 5 rows

df.head(10)                         # First 10 rows

df.tail()                           # Last 5 rows

df.tail(3)                          # Last 3 rows

  

# Shape and dimensions

df.shape                            # (rows, columns)

df.height                           # Number of rows

df.width                            # Number of columns

  

# Column and schema information

df.columns                          # Column names

df.dtypes                           # Data types of columns

df.schema                           # Schema (dtype per column)

df.null_count()                     # Count null values per column

df.estimated_size()                 # Estimated memory usage

  

# Descriptive statistics

df.describe()                       # Statistical summary

df.describe(percentiles=[0.25, 0.5, 0.75])  # Custom percentiles

  

# Value counts

df['column'].value_counts()         # Frequency of unique values

df['column'].value_counts(sort=True)  # Sorted by count

#### Data Manipulation Functions

python

# Selecting columns

df.select('column')                 # Single column as DataFrame

df.select(['col1', 'col2'])         # Multiple columns

df[['col1', 'col2']]                # Multiple columns (shorthand)

  

# Selecting rows

df.filter(pl.col('age') > 25)       # Rows where age > 25

df.filter((pl.col('age') > 25) & (pl.col('salary') > 50000))  # Multiple conditions

  

# Query method (more readable)

df.filter(pl.col('age') > 25).select(['name', 'salary'])

  

# Adding/Modifying columns

df.with_columns([

    (pl.col('col1') + pl.col('col2')).alias('new_column'),

    pl.col('col1').pow(2).alias('squared')

])

  

# Removing columns

df.drop('column')                   # Drop column

df.drop(['col1', 'col2'])           # Drop multiple columns

  

# Renaming columns

df.rename({'old_name': 'new_name'})

df.rename({'old1': 'new1', 'old2': 'new2'})  # Multiple columns

#### Data Cleaning Functions

python

# Handling null values

df.drop_nulls()                     # Drop rows with any null

df.drop_nulls(subset=['col1', 'col2'])  # Drop rows with null in specific columns

df.fill_null(0)                     # Fill null with 0

df.fill_null(strategy='forward')    # Forward fill

df.fill_null(strategy='backward')   # Backward fill

df.fill_null(pl.col('col').mean())  # Fill with mean value

  

# Handling duplicates

df.is_duplicated()                  # Boolean mask of duplicates

df.unique()                         # Remove duplicate rows

df.unique(subset=['col1'])          # Remove duplicates in specific columns

  

# Data type conversion

df.cast({'col': pl.Float64})        # Convert column to float

df.cast({'col': pl.Int64})          # Convert column to int

df.cast({'col': pl.Utf8})           # Convert column to string

df.cast({'col': pl.Datetime})       # Convert to datetime

df.cast({'col': pl.Date})           # Convert to date

  

# String operations

df.with_columns([

    pl.col('text').str.to_lowercase().alias('lowercase'),

    pl.col('text').str.to_uppercase().alias('uppercase'),

    pl.col('text').str.strip_chars().alias('stripped'),

    pl.col('text').str.contains('pattern').alias('contains_pattern'),

    pl.col('text').str.extract('(\\d+)').alias('extracted')

])

#### Data Aggregation Functions

python

# Basic aggregation

df.select([

    pl.col('col').mean(),           # Mean

    pl.col('col').sum(),            # Sum

    pl.col('col').min(),            # Minimum

    pl.col('col').max(),            # Maximum

    pl.col('col').median(),         # Median

    pl.col('col').std(),            # Standard deviation

    pl.col('col').var(),            # Variance

    pl.col('col').n_unique()        # Number of unique values

])

  

# Groupby operations

df.group_by('category').agg([

    pl.col('value').mean().alias('avg_value'),

    pl.col('value').std().alias('std_value'),

    pl.col('value').count().alias('count')

])

  

# Pivot operations

df.pivot(index='category', 

         columns='year', 

         values='value', 

         aggregate_function='mean')

#### Merging and Joining Functions

python

# Joining DataFrames

df1.join(df2, on='key')                     # Inner join on key

df1.join(df2, on='key', how='left')         # Left join

df1.join(df2, on='key', how='right')        # Right join

df1.join(df2, on='key', how='outer')        # Outer join

df1.join(df2, on='key', how='semi')         # Semi join (only left columns)

df1.join(df2, left_on='key1', right_on='key2')  # Different key columns

  

# Concatenating DataFrames

pl.concat([df1, df2])                       # Row-wise concatenation

pl.concat([df1, df2], how='vertical')       # Row-wise (explicit)

pl.concat([df1, df2], how='horizontal')     # Column-wise concatenation

pl.concat([df1, df2], how='diagonal')       # Fill missing columns with null

  

# Joining on multiple columns

df1.join(df2, on=['key1', 'key2'])

#### Time Series Functions

python

# Date and time operations

pl.date_range(start='2024-01-01', end='2024-12-31', interval='1d')  # Daily dates

pl.date_range(start='2024-01-01', periods=10, interval='1mo')        # Monthly dates

  

# Extract date components

df.with_columns([

    pl.col('date').dt.year().alias('year'),       # Extract year

    pl.col('date').dt.month().alias('month'),     # Extract month

    pl.col('date').dt.day().alias('day'),         # Extract day

    pl.col('date').dt.weekday().alias('weekday'), # Extract weekday

    pl.col('date').dt.quarter().alias('quarter')  # Extract quarter

])

  

# Resampling (group by time intervals)

df.group_by_dynamic('date', every='1d').agg([

    pl.col('value').mean().alias('daily_avg')

])

  

# Shifting and lagging

df.with_columns([

    pl.col('col').shift(1).alias('lag_1'),       # Lag by 1 period

    pl.col('col').shift(-1).alias('lead_1'),     # Lead by 1 period

    pl.col('col').diff().alias('diff'),          # Difference between periods

    pl.col('col').pct_change().alias('pct_change')  # Percentage change

])

### Lazy Evaluation in Polars

python

# Lazy evaluation: build a query plan, execute only when needed

lazy_df = pl.scan_csv('large_data.csv')          # Scan CSV lazily

lazy_df = lazy_df.filter(pl.col('age') > 25)     # Add filter

lazy_df = lazy_df.group_by('category').agg(      # Add groupby

    pl.col('value').mean()

)

lazy_df = lazy_df.sort('value', descending=True) # Add sort

df = lazy_df.collect()                           # Execute and collect

  

# Benefits of lazy evaluation:

# 1. Query optimization

# 2. Pushdown filters to the data source

# 3. Projection pushdown (select only needed columns)

# 4. Pipeline parallelism

### Common Mistakes with Polars Imports

1. Mixing Pandas and Polars syntax

python

# Wrong - using Pandas syntax with Polars

df['age'] > 25  # This creates a boolean Series, not a Polars expression

  

# Correct - use Polars expressions

df.filter(pl.col('age') > 25)

2. Forgetting .collect() with lazy operations

python

# Wrong - lazy frame is not executed

lazy_df = pl.scan_csv('data.csv')

result = lazy_df.filter(pl.col('age') > 25)  # LazyFrame, not DataFrame

print(result)  # Prints LazyFrame, not the data

  

# Correct - use .collect()

result = lazy_df.filter(pl.col('age') > 25).collect()

3. Using .loc or .iloc (not available in Polars)

python

# Wrong

df.loc[0]      # AttributeError

df.iloc[0]     # AttributeError

  

# Correct

df[0, :]       # Get first row (when using indexing)

df.row(0)      # Get first row as tuple

Interview Note: Interviewers are increasingly asking about Polars as a modern alternative to Pandas. Be prepared to discuss performance differences, lazy evaluation, and the Arrow memory model.

Best Practice: For new projects, consider Polars when working with large datasets or when performance is critical. However, maintain compatibility by using both libraries when needed (via pl.from_pandas() and pl.to_pandas()).

---

## 3.5 DuckDB Imports

### Core Import

python

import duckdb

What it imports: The entire DuckDB library, which provides an embedded SQL database engine with strong integration with Python.

Why it's used: DuckDB is designed to run complex analytical SQL queries directly on data in memory or on disk, without the need for a separate database server. It offers:

- SQL power: Full SQL support with advanced analytics
    
- In-memory processing: Fast query execution
    
- Multiple data sources: Query CSV, Parquet, JSON, and Pandas/Polars DataFrames
    
- Out-of-core processing: Handle datasets larger than memory
    
- Excellent performance: Compiled with vectorized execution
    

When to use:

- When you prefer SQL over Python for data transformations
    
- When working with datasets that don't fit in memory
    
- When you need to query data directly from files without loading into memory
    
- When performing complex joins and aggregations on large datasets
    

When to avoid:

- When you need to do complex data transformations that are easier in Python
    
- When you need advanced machine learning capabilities
    
- When you need a fully-featured database with user management
    

### Essential Sub-Imports

python

# DuckDB doesn't have many sub-imports, but you can import specific functions

from duckdb import DuckDBPyConnection, DuckDBPyRelation

  

# For convenience, you can import as:

import duckdb as db

### Frequently Used Functions and Their Imports

#### Creating Connections and Querying

python

import duckdb

  

# In-memory connection (default)

conn = duckdb.connect()

# Or simply:

conn = duckdb.connect(':memory:')

  

# Persistent database

conn = duckdb.connect('my_database.db')

  

# Default connection (creates temporary in-memory)

conn = duckdb.default_connection

  

# Simple queries

result = duckdb.sql("SELECT 1 + 1")  # Returns a relation

print(result.fetchall())  # [(2,)]

  

# Execute queries

duckdb.execute("CREATE TABLE numbers (id INTEGER, value INTEGER)")

duckdb.execute("INSERT INTO numbers VALUES (1, 10), (2, 20), (3, 30)")

duckdb.execute("SELECT * FROM numbers").fetchall()  # [(1, 10), (2, 20), (3, 30)]

#### Working with DataFrames

python

import pandas as pd

import polars as pl

  

# Query Pandas DataFrame

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

result = duckdb.sql("SELECT * FROM df WHERE a > 1")

print(result.df())  # Convert to pandas DataFrame

  

# Query Polars DataFrame

pl_df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

result = duckdb.sql("SELECT * FROM pl_df WHERE a > 1")

print(result.pl())  # Convert to polars DataFrame

  

# Register DataFrames

duckdb.register('my_df', df)

result = duckdb.sql("SELECT * FROM my_df")

#### Reading External Data

python

# Read CSV files

duckdb.sql("SELECT * FROM read_csv('data.csv')")

duckdb.sql("""

    SELECT * FROM read_csv('data.csv', 

        sep=',',

        header=True,

        columns={'col1': 'int', 'col2': 'varchar'},

        auto_detect=False

    )

""")

  

# Read Parquet files

duckdb.sql("SELECT * FROM read_parquet('data.parquet')")

duckdb.sql("SELECT * FROM 'data.parquet'")  # Shorthand

  

# Read JSON files

duckdb.sql("SELECT * FROM read_json('data.json')")

duckdb.sql("SELECT * FROM read_ndjson('data.ndjson')")

  

# Read multiple files (glob pattern)

duckdb.sql("SELECT * FROM read_parquet('data/*.parquet')")

  

# Read from URL

duckdb.sql("SELECT * FROM read_csv('https://example.com/data.csv')")

#### Data Manipulation

python

# Create tables

conn.execute("""

    CREATE TABLE employees AS 

    SELECT * FROM read_csv('employees.csv')

""")

  

# Insert data

conn.execute("INSERT INTO employees VALUES (?, ?, ?)", [1, 'Alice', 'IT'])

conn.executemany("INSERT INTO employees VALUES (?, ?, ?)", 

                 [(2, 'Bob', 'HR'), (3, 'Charlie', 'Finance')])

  

# Pivot/Cross-tab

duckdb.sql("""

    PIVOT sales ON product USING SUM(amount)

""")

  

# Window functions

duckdb.sql("""

    SELECT *,

           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank,

           SUM(salary) OVER (PARTITION BY department) as dept_total

    FROM employees

""")

  

# JSON operations

duckdb.sql("""

    SELECT 

        json_extract(data, '$.name') as name,

        json_extract(data, '$.age') as age

    FROM json_table

""")

#### Advanced Analytics

python

# Statistical functions

duckdb.sql("""

    SELECT 

        AVG(salary) as mean,

        STDDEV(salary) as stddev,

        MIN(salary) as min,

        MAX(salary) as max,

        QUANTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) as median

    FROM employees

""")

  

# Time series

duckdb.sql("""

    SELECT 

        DATE_TRUNC('month', order_date) as month,

        SUM(amount) as total_sales

    FROM orders

    GROUP BY month

    ORDER BY month

""")

  

# Text search

duckdb.sql("""

    SELECT * FROM articles 

    WHERE content ILIKE '%machine learning%'

""")

### Working with Large Datasets

python

# Query large Parquet files without loading into memory

result = duckdb.sql("""

    SELECT 

        category,

        SUM(sales) as total_sales,

        AVG(price) as avg_price

    FROM 'sales_*.parquet'

    GROUP BY category

    ORDER BY total_sales DESC

""")

  

# Use temporary views for complex queries

conn.execute("""

    CREATE OR REPLACE TEMP VIEW sales_stats AS

    SELECT 

        category,

        SUM(sales) as total,

        COUNT(*) as count

    FROM 'sales_*.parquet'

    GROUP BY category

""")

  

result = conn.execute("SELECT * FROM sales_stats WHERE total > 1000")

### Common Mistakes with DuckDB Imports

1. Forgetting to convert to Pandas/Polars

python

# Wrong - result is a DuckDB relation, not a DataFrame

result = duckdb.sql("SELECT * FROM df")

print(result.head())  # AttributeError

  

# Correct - convert to DataFrame

result_df = duckdb.sql("SELECT * FROM df").df()  # Pandas

result_pl = duckdb.sql("SELECT * FROM df").pl()  # Polars

2. Mixing SQL and Python incorrectly

python

# Wrong - Python variable not interpolated

col_name = 'salary'

duckdb.sql("SELECT * FROM employees WHERE col_name > 50000")  # Error

  

# Correct - use Python string formatting

duckdb.sql(f"SELECT * FROM employees WHERE {col_name} > 50000")

  

# Better - use parameterized queries

duckdb.sql("SELECT * FROM employees WHERE salary > ?", [50000])

3. Not using file glob patterns correctly

python

# Wrong - only reads one file

duckdb.sql("SELECT * FROM read_parquet('data/file1.parquet')")

  

# Correct - reads all matching files

duckdb.sql("SELECT * FROM read_parquet('data/*.parquet')")

Interview Note: DuckDB is increasingly popular for data engineering roles. Interviewers might ask about its advantages over traditional databases or when to use it versus Pandas.

Best Practice: Use DuckDB for complex SQL queries and when working with larger-than-memory datasets. It's especially useful in data pipelines for initial data exploration and transformation.

---

## 3.6 SciPy Imports

### Core Import

python

import scipy as sp

What it imports: The entire SciPy library, which builds on NumPy and provides scientific and technical computing tools.

Why it's used: SciPy offers advanced mathematical functions, algorithms, and utilities for:

- Optimization
    
- Integration
    
- Interpolation
    
- Signal processing
    
- Image processing
    
- Sparse matrix operations
    
- Scientific computing
    

When to use:

- When you need advanced mathematical functions not in NumPy
    
- For scientific computing applications
    
- When doing signal or image processing
    

When to avoid:

- For basic array operations (use NumPy)
    
- For machine learning (use Scikit-learn)
    
- For statistical analysis (use Statsmodels)
    

### Essential Sub-Imports

python

# Statistics module

from scipy import stats

  

# Optimization

from scipy import optimize

  

# Linear algebra (extends NumPy's linalg)

from scipy import linalg

  

# Integration

from scipy import integrate

  

# Interpolation

from scipy import interpolate

  

# Signal processing

from scipy import signal

  

# Image processing

from scipy import ndimage

  

# Sparse matrices

from scipy import sparse

from scipy.sparse import csr_matrix, csc_matrix

  

# Spatial data structures

from scipy import spatial

### Frequently Used Functions and Their Imports

#### Statistics Module (scipy.stats)

python

from scipy import stats

  

# Distribution functions

norm = stats.norm                  # Normal distribution

norm.pdf(0)                        # PDF at 0

norm.cdf(0)                        # CDF at 0

norm.ppf(0.975)                    # Percent point function (z-score)

norm.rvs(100)                      # Generate 100 random samples

  

# Statistical tests

# t-test

t_stat, p_value = stats.ttest_ind(group1, group2)  # Independent t-test

t_stat, p_value = stats.ttest_rel(before, after)   # Paired t-test

t_stat, p_value = stats.ttest_1samp(data, popmean) # One-sample t-test

  

# ANOVA

f_stat, p_value = stats.f_oneway(group1, group2, group3)

  

# Chi-square test

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

  

# Correlation tests

corr, p_value = stats.pearsonr(x, y)    # Pearson correlation

corr, p_value = stats.spearmanr(x, y)   # Spearman correlation

corr, p_value = stats.kendalltau(x, y)  # Kendall's tau

  

# Normality tests

stat, p_value = stats.shapiro(data)     # Shapiro-Wilk test

stat, p_value = stats.normaltest(data)  # D'Agostino's K^2 test

  

# Descriptive statistics

mean = stats.tmean(data)                # Trimmed mean

variance = stats.tvar(data)             # Trimmed variance

skewness = stats.skew(data)             # Skewness

kurtosis = stats.kurtosis(data)         # Kurtosis

mode = stats.mode(data)                 # Mode

  

# Z-score and others

z_scores = stats.zscore(data)           # Z-scores

descriptive = stats.describe(data)      # Comprehensive description

#### Optimization Module (scipy.optimize)

python

from scipy import optimize

  

# Minimization of scalar functions

result = optimize.minimize_scalar(fun, bounds=(0, 10), method='bounded')

result.x  # Minimum point

result.fun  # Minimum value

  

# Minimization of multi-variable functions

result = optimize.minimize(

    fun,                           # Function to minimize

    x0,                            # Initial guess

    method='BFGS',                 # Optimization method

    jac=gradient_function,         # Optional gradient

    hess=hessian_function,         # Optional Hessian

    constraints=constraints        # Optional constraints

)

  

# Curve fitting

popt, pcov = optimize.curve_fit(

    func,                          # Model function

    xdata, ydata,                  # Data

    p0=[1, 1, 1]                   # Initial guess for parameters

)

  

# Root finding

root = optimize.root(func, x0)      # Find root of system

root = optimize.bisect(func, a, b)  # Use bisection method

  

# Linear programming

result = optimize.linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq)

#### Linear Algebra Module (scipy.linalg)

python

from scipy import linalg

  

# Solve linear systems

x = linalg.solve(A, b)              # Solve Ax = b

x = linalg.lstsq(A, b)              # Least squares solution

  

# Matrix decompositions

P, L, U = linalg.lu(A)              # LU decomposition

Q, R = linalg.qr(A)                 # QR decomposition

U, S, Vh = linalg.svd(A)            # SVD decomposition

eigvals, eigvecs = linalg.eig(A)    # Eigenvalue decomposition

L = linalg.cholesky(A)              # Cholesky decomposition

  

# Matrix norms

norm = linalg.norm(A)               # Frobenius norm

norm = linalg.norm(A, ord=1)        # L1 norm

norm = linalg.norm(A, ord=2)        # L2 norm

  

# Matrix functions

expm = linalg.expm(A)               # Matrix exponential

logm = linalg.logm(A)               # Matrix logarithm

sqrtm = linalg.sqrtm(A)             # Matrix square root

#### Integration Module (scipy.integrate)

python

from scipy import integrate

  

# Definite integral

result, error = integrate.quad(func, a, b)  # 1D integral

result, error = integrate.dblquad(func, a, b, gfun, hfun)  # 2D integral

result, error = integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun)  # 3D integral

  

# ODE solver

from scipy.integrate import odeint

  

def model(y, t):

    dydt = -y + 1

    return dydt

  

y0 = 0

t = np.linspace(0, 10, 100)

solution = odeint(model, y0, t)

#### Interpolation Module (scipy.interpolate)

python

from scipy import interpolate

  

# 1D interpolation

f = interpolate.interp1d(x, y, kind='cubic')  # Cubic spline

f = interpolate.interp1d(x, y, kind='linear') # Linear interpolation

y_new = f(x_new)

  

# Spline interpolation

tck = interpolate.splrep(x, y, s=0)  # Create spline

y_new = interpolate.splev(x_new, tck)  # Evaluate spline

  

# 2D interpolation

f = interpolate.interp2d(x, y, z, kind='cubic')

z_new = f(x_new, y_new)

  

# Grid data interpolation

from scipy.interpolate import griddata

grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

### Common Mistakes with SciPy Imports

1. Importing incorrectly

python

# Wrong - too general

from scipy import *  # Could cause namespace conflicts

  

# Better - import specific modules

from scipy import stats

2. Forgetting that SciPy functions often return tuples

python

# Wrong - trying to index incorrectly

result = stats.ttest_ind(group1, group2)

print(result[0])  # OK, but not clear

  

# Better - unpack properly

t_stat, p_value = stats.ttest_ind(group1, group2)

3. Not checking for NaN or inf values

python

# Wrong - SciPy functions may fail with NaN

data = [1, 2, np.nan, 4, 5]

stats.describe(data)  # May return unexpected results

  

# Correct - handle missing values first

clean_data = data[~np.isnan(data)]

stats.describe(clean_data)

Interview Note: SciPy is often mentioned in interviews for data science roles, particularly for the statistics module. Know the common statistical tests and when to use them.

Best Practice: Use SciPy when you need advanced mathematical functions. For basic statistics, NumPy often suffices. For specialized statistical modeling, use Statsmodels.

---

## 3.7 Statsmodels Imports

### Core Import

python

import statsmodels.api as sm

What it imports: The Statsmodels library for statistical modeling, hypothesis testing, and exploratory data analysis.

Why it's used: Statsmodels provides:

- Classical statistical models (OLS regression, GLM)
    
- Time series analysis (ARIMA, VAR)
    
- Hypothesis tests and confidence intervals
    
- R-like formula interface
    
- Detailed statistical output with p-values, confidence intervals, etc.
    

When to use:

- When you need inferential statistics (p-values, confidence intervals)
    
- For statistical hypothesis testing
    
- For time series forecasting
    
- For econometric analysis
    
- When you need model diagnostics and summary statistics
    

When to avoid:

- For machine learning prediction (use Scikit-learn)
    
- For deep learning
    
- For simple statistical summaries (use Pandas or NumPy)
    

### Essential Sub-Imports

python

# Main API (most commonly used)

import statsmodels.api as sm

  

# Formula API (R-like syntax)

import statsmodels.formula.api as smf

  

# Time series analysis

import statsmodels.tsa.api as tsa

  

# Robust statistics

import statsmodels.robust.api as smrobust

  

# Graphics (additional plotting functions)

import statsmodels.graphics.api as smgraph

  

# Additive models (GAM)

import statsmodels.gam.api as smgam

### Frequently Used Functions and Their Imports

#### Linear Regression

python

import statsmodels.api as sm

import statsmodels.formula.api as smf

import pandas as pd

  

# OLS with formula interface

model = smf.ols(formula='y ~ x1 + x2', data=df).fit()

print(model.summary())  # Comprehensive output

print(model.params)     # Coefficients

print(model.pvalues)    # P-values

print(model.conf_int()) # Confidence intervals

  

# OLS with API (adds intercept automatically)

X = df[['x1', 'x2']]    # Predictors

X = sm.add_constant(X)  # Add intercept

y = df['y']             # Response

model = sm.OLS(y, X).fit()

  

# Weighted Least Squares (WLS)

weights = 1 / (df['x1'] ** 2)

model = sm.WLS(y, X, weights=weights).fit()

  

# Generalized Least Squares (GLS)

model = sm.GLS(y, X, cov_type='HC3').fit()

  

# Quantile Regression

model = sm.QuantReg(y, X).fit(q=0.5)  # Median

#### Generalized Linear Models (GLM)

python

# Logistic regression

model = smf.glm(

    formula='y ~ x1 + x2',

    data=df,

    family=sm.families.Binomial()

).fit()

  

# Poisson regression

model = smf.glm(

    formula='count ~ x1 + x2',

    data=df,

    family=sm.families.Poisson()

).fit()

  

# Gamma regression

model = smf.glm(

    formula='y ~ x1 + x2',

    data=df,

    family=sm.families.Gamma()

).fit()

#### Time Series Analysis

python

import statsmodels.api as sm

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.stattools import adfuller, acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

  

# Stationarity test

result = adfuller(series)

print(f'ADF Statistic: {result[0]:.4f}')

print(f'p-value: {result[1]:.4f}')

  

# Autocorrelation

acf_values = acf(series, nlags=20)

pacf_values = pacf(series, nlags=20)

  

# Seasonal decomposition

decomposition = seasonal_decompose(series, model='additive', period=12)

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid

  

# ARIMA model

model = ARIMA(series, order=(p, d, q)).fit()

forecast = model.forecast(steps=10)

print(model.summary())

  

# VAR (Vector Autoregression)

from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(df[['y1', 'y2']]).fit(maxlags=5)

forecast = model.forecast(model.endog[-5:], steps=10)

#### Model Diagnostics

python

# Residual diagnostics

residuals = model.resid

predicted = model.fittedvalues

  

# Plot residuals

import matplotlib.pyplot as plt

plt.scatter(predicted, residuals)

plt.axhline(y=0, color='r', linestyle='--')

plt.xlabel('Fitted values')

plt.ylabel('Residuals')

plt.show()

  

# QQ plot for normality

sm.qqplot(residuals, line='s')

plt.show()

  

# Durbin-Watson test for autocorrelation

from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)

  

# Breusch-Pagan test for heteroscedasticity

from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(residuals, X)

  

# Variance Inflation Factor (VIF) for multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#### Hypothesis Testing

python

# t-test

t_test_result = model.t_test('x1 = 0')  # Test coefficient

t_test_result = model.t_test('x1 = x2')  # Test equality of coefficients

  

# F-test

f_test_result = model.f_test('x1 = 0, x2 = 0')  # Joint test

f_test_result = model.f_test('x1 = 0, x2 = x3')  # Mixed hypothesis

  

# Wald test

wald_test = model.wald_test('x1 = x2, x3 = 0')

  

# Log-likelihood ratio test

from statsmodels.stats.anova import anova_lm

anova_result = anova_lm(model1, model2)  # Compare nested models

#### Statistical Tests

python

from statsmodels.stats import weightstats, proportion

  

# Z-test for means

z_stat, p_value = weightstats.ztest(data1, data2, value=0)

z_stat, p_value = weightstats.ztest(data, value=0)

  

# Proportion tests

stat, p_value = proportion.proportions_ztest(count, nobs, value=0.5)

stat, p_value = proportion.proportions_chisquare(count, nobs)

  

# Normality tests

from statsmodels.stats.diagnostic import lilliefors

stat, p_value = lilliefors(data)  # Lilliefors test (Kolmogorov-Smirnov)

  

# Non-parametric tests

from scipy import stats

stat, p_value = stats.mannwhitneyu(data1, data2)  # Mann-Whitney U test

stat, p_value = stats.wilcoxon(data1, data2)      # Wilcoxon signed-rank test

stat, p_value = stats.kruskal(data1, data2, data3)  # Kruskal-Wallis test

### Common Mistakes with Statsmodels Imports

1. Forgetting to add intercept

python

# Wrong - no intercept

X = df[['x1', 'x2']]  # OLS without intercept

model = sm.OLS(y, X).fit()  # This forces regression through origin

  

# Correct - add intercept

X = sm.add_constant(df[['x1', 'x2']])

model = sm.OLS(y, X).fit()

2. Using formula API incorrectly

python

# Wrong

model = smf.ols('y ~ x1 + x2', data=df).fit()  # Variable names must match

  

# Correct - use actual column names from df

model = smf.ols('sales ~ price + advertising', data=df).fit()

3. Not handling categorical variables

python

# Wrong - treating categorical as numeric

model = smf.ols('y ~ category', data=df).fit()  # Will treat as numeric

  

# Correct - create dummy variables

model = smf.ols('y ~ C(category)', data=df).fit()

model = smf.ols('y ~ C(category, Treatment(reference="A"))', data=df).fit()

Interview Note: Statsmodels is crucial for understanding the statistical basis of models. Interviewers often ask about interpreting regression outputs, p-values, and model diagnostics.

Best Practice: Use Statsmodels when you need to understand relationships and make inferences. For prediction-focused work, consider Scikit-learn. For time series, Statsmodels is often preferred over other libraries.

---

## 3.8 Matplotlib Imports

### Core Import

python

import matplotlib.pyplot as plt

What it imports: The pyplot module from Matplotlib, which provides a MATLAB-like plotting interface.

Why it's used: Matplotlib is the fundamental plotting library in Python. It provides:

- Complete control over every aspect of a figure
    
- Publication-quality graphics
    
- Support for multiple backends
    
- Integration with Pandas and NumPy
    
- Customizable styles and themes
    

When to use:

- When you need fine-grained control over plots
    
- For creating publication-ready figures
    
- When building complex multi-panel figures
    
- As a foundation for other visualization libraries
    

When to avoid:

- For quick exploratory analysis (use Seaborn or Plotly)
    
- When you need interactivity (use Plotly)
    
- For statistical visualizations (use Seaborn)
    

### Essential Sub-Imports

python

# Main plotting interface

import matplotlib.pyplot as plt

  

# For creating subplots

from matplotlib import gridspec

  

# For customizing styles

import matplotlib as mpl

from matplotlib import style

  

# For animations

import matplotlib.animation as animation

  

# For colormaps

from matplotlib import cm

  

# For font management

from matplotlib import font_manager

  

# For patches and shapes

from matplotlib import patches

from matplotlib.patches import Rectangle, Circle, Arrow

  

# For 3D plotting

from mpl_toolkits.mplot3d import Axes3D

### Frequently Used Functions and Their Imports

#### Basic Plotting

python

import matplotlib.pyplot as plt

import numpy as np

  

# Line plot

x = np.linspace(0, 10, 100)

y = np.sin(x)

plt.plot(x, y)

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.title('Sine Wave')

plt.grid(True)

plt.show()

  

# Scatter plot

plt.scatter(x, y, alpha=0.5, color='blue', s=50)

  

# Bar plot

categories = ['A', 'B', 'C', 'D']

values = [10, 15, 7, 12]

plt.bar(categories, values, color=['red', 'green', 'blue', 'orange'])

  

# Histogram

data = np.random.normal(0, 1, 1000)

plt.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')

  

# Box plot

plt.boxplot([np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)])

  

# Error bars

plt.errorbar(x, y, yerr=0.1, fmt='o', capsize=5)

#### Figure and Subplot Management

python

# Create figure with specific size

fig = plt.figure(figsize=(10, 6))

  

# Create subplots

fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid

axes[0, 0].plot(x, y)

axes[0, 1].hist(data)

axes[1, 0].scatter(x, y)

axes[1, 1].bar(categories, values)

plt.tight_layout()  # Adjust spacing

  

# GridSpec for complex layouts

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))

gs = gridspec.GridSpec(3, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns

ax2 = fig.add_subplot(gs[1:, 0])  # Middle row, first column

ax3 = fig.add_subplot(gs[1:, 1:])  # Middle row, remaining columns

#### Customization Functions

python

# Line styles and markers

plt.plot(x, y, 'r--', linewidth=2, markersize=8)  # Red dashed line

plt.plot(x, y, 'bo-', label='Data')  # Blue circles with line

plt.plot(x, y, linestyle='dashed', color='green', marker='s', mfc='red')

  

# Colors and colormaps

colors = plt.cm.viridis(np.linspace(0, 1, 10))  # Generate colors

plt.scatter(x, y, c=data, cmap='viridis', s=50)  # Colormap mapping

  

# Text and annotations

plt.text(5, 0.5, 'Important point', fontsize=12, bbox=dict(facecolor='yellow'))

plt.annotate('Max value', xy=(1.5, 1), xytext=(3, 1.5), 

             arrowprops=dict(facecolor='black', shrink=0.05))

  

# Legend

plt.legend(['Data 1', 'Data 2'], loc='upper right', fontsize=10, framealpha=0.5)

  

# Ticks and labels

plt.xticks([0, 2, 4, 6, 8, 10], ['Zero', 'Two', 'Four', 'Six', 'Eight', 'Ten'])

plt.xticks(rotation=45)

plt.yticks(fontsize=12)

#### Saving Figures

python

# Save figure in multiple formats

plt.savefig('plot.png', dpi=300, bbox_inches='tight', transparent=True)

plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')

plt.savefig('plot.svg', format='svg')

  

# Save with tight layout to avoid clipping

fig.savefig('figure.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

#### Style Customization

python

# Use built-in styles

plt.style.use('ggplot')

plt.style.use('seaborn-v0_8-darkgrid')

plt.style.use('fivethirtyeight')

  

# Create custom style

plt.rcParams['font.size'] = 14

plt.rcParams['axes.labelsize'] = 16

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['legend.fontsize'] = 12

plt.rcParams['figure.titlesize'] = 18

  

# Custom style dictionary

plt.rcParams.update({

    'axes.grid': True,

    'grid.linestyle': '--',

    'grid.alpha': 0.5,

    'lines.linewidth': 2,

    'lines.markersize': 8

})

  

# Reset to default

plt.rcParams.update(plt.rcParamsDefault)

#### Specialized Plot Types

python

# 3D plots

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=z, cmap='viridis')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

  

# Contour plots

plt.contour(X, Y, Z, levels=20, cmap='plasma')

plt.contourf(X, Y, Z, levels=20, cmap='plasma')  # Filled contour

  

# Heatmap

plt.imshow(data, cmap='coolwarm', aspect='auto')

plt.colorbar(label='Values')

  

# Image display

plt.imshow(image_array, cmap='gray')

plt.axis('off')

  

# Filled area

plt.fill_between(x, y1, y2, alpha=0.3, color='blue')

plt.fill_betweenx(y, x1, x2, alpha=0.3, color='red')

  

# Stacked plots

plt.stackplot(x, data1, data2, data3, labels=['A', 'B', 'C'])

### Common Mistakes with Matplotlib Imports

1. Not using plt.close()

python

# Wrong - memory accumulation

for i in range(100):

    plt.plot(x, y)

    plt.savefig(f'plot_{i}.png')  # Memory leak!

  

# Correct - close figure after saving

for i in range(100):

    fig, ax = plt.subplots()

    ax.plot(x, y)

    fig.savefig(f'plot_{i}.png')

    plt.close(fig)

2. Mixing OOP and pyplot interfaces

python

# Confusing - mixing styles

fig, ax = plt.subplots()

ax.plot(x, y)

plt.xlabel('X')  # Works but confusing

  

# Better - stick to one style

# Option 1: pyplot (simple)

plt.plot(x, y)

plt.xlabel('X')

  

# Option 2: OOP (more control)

fig, ax = plt.subplots()

ax.plot(x, y)

ax.set_xlabel('X')

3. Forgetting to call plt.show()

python

# Wrong - no display

plt.plot(x, y)  # Nothing appears

  

# Correct - display the plot

plt.plot(x, y)

plt.show()

  

# For Jupyter notebooks, use:

%matplotlib inline  # Display inline

Interview Note: Matplotlib is often tested in interviews through coding exercises. Know how to create basic plots, customize them, and create multi-panel figures.

Best Practice: Use the OOP interface (fig, ax = plt.subplots()) for complex figures and when you need more control. Use the pyplot interface for quick exploratory plots. Always use plt.close() when generating many figures.

---

## 3.9 Seaborn Imports

### Core Import

python

import seaborn as sns

What it imports: The entire Seaborn library, which provides a high-level interface for drawing statistical graphics.

Why it's used: Seaborn builds on Matplotlib to create beautiful, informative statistical plots with minimal code. It offers:

- Statistical visualizations (distributions, regressions, categorical plots)
    
- Built-in themes and color palettes
    
- Integration with Pandas DataFrames
    
- Multi-plot grids
    
- Support for complex statistical analyses
    

When to use:

- For exploratory data analysis
    
- When creating statistical visualizations
    
- When you want attractive plots with minimal code
    
- For data exploration in EDA
    

When to avoid:

- When you need fine-grained control over every aspect
    
- For production-ready publication figures (use Matplotlib for final adjustments)
    
- For custom visualizations not covered by Seaborn
    

### Essential Sub-Imports

python

# Main library (no sub-imports needed)

import seaborn as sns

  

# But you can import specific functions

from seaborn import set_style, set_palette, despine

  

# For color palettes

from seaborn import color_palette, set_palette

### Frequently Used Functions and Their Imports

#### Setting Styles and Themes

python

import seaborn as sns

import matplotlib.pyplot as plt

  

# Set style

sns.set_style('whitegrid')

sns.set_style('darkgrid')

sns.set_style('white')

sns.set_style('dark')

sns.set_style('ticks')

  

# Set context (scale of elements)

sns.set_context('paper')

sns.set_context('notebook')

sns.set_context('talk')

sns.set_context('poster')

  

# Set palette

sns.set_palette('viridis')

sns.set_palette('Set2')

sns.set_palette('coolwarm')

sns.set_palette('husl')

  

# Custom palette

palette = sns.color_palette('viridis', 8)

sns.set_palette(palette)

  

# Reset defaults

sns.reset_defaults()

  

# Remove spines

sns.despine()

sns.despine(left=True, bottom=True, top=True, right=True)

#### Distribution Plots

python

# Histogram with KDE

sns.histplot(data, bins=30, kde=True)

  

# Density plot

sns.kdeplot(data, shade=True)

sns.kdeplot(data1, label='Group 1')

sns.kdeplot(data2, label='Group 2')

  

# Rug plot (shows individual data points)

sns.rugplot(data, height=0.1)

  

# Combined distribution plot

sns.displot(data, kind='hist', kde=True, rug=True)

  

# Box plot with distribution

sns.boxplot(x='category', y='value', data=df)

sns.violinplot(x='category', y='value', data=df)

sns.boxenplot(x='category', y='value', data=df)  # More compact box plot

  

# Pair plot (all pairwise relationships)

sns.pairplot(df, hue='category', diag_kind='kde')

#### Regression Plots

python

# Regression line with confidence interval

sns.regplot(x='x', y='y', data=df)

sns.regplot(x='x', y='y', data=df, lowess=True)  # Locally weighted regression

  

# Residual plot

sns.residplot(x='x', y='y', data=df)

  

# Regression with categorical variables

sns.lmplot(x='x', y='y', data=df, hue='category')

sns.lmplot(x='x', y='y', data=df, col='category')  # Separate plots per category

#### Categorical Plots

python

# Bar plot

sns.barplot(x='category', y='value', data=df)

sns.barplot(x='category', y='value', data=df, hue='group')

  

# Point plot

sns.pointplot(x='category', y='value', data=df)

sns.pointplot(x='category', y='value', data=df, hue='group')

  

# Count plot (frequency)

sns.countplot(x='category', data=df)

sns.countplot(x='category', hue='group', data=df)

  

# Strip plot (jittered scatter plot)

sns.stripplot(x='category', y='value', data=df, jitter=True)

  

# Swarm plot (points spread out to show distribution)

sns.swarmplot(x='category', y='value', data=df)

  

# Catplot (flexible categorical plot)

sns.catplot(x='category', y='value', data=df, kind='box')

sns.catplot(x='category', y='value', data=df, kind='violin')

#### Matrix Plots

python

# Heatmap

corr = df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)

sns.heatmap(corr, annot=True, cmap='RdYlBu_r', vmin=-1, vmax=1)

  

# Cluster map (heatmap with clustering)

sns.clustermap(corr, annot=True, cmap='coolwarm')

  

# Pair grid (customized pair plot)

g = sns.PairGrid(df, hue='category')

g.map_upper(sns.scatterplot)

g.map_lower(sns.kdeplot)

g.map_diag(sns.histplot)

#### Time Series Plots

python

# Line plot with confidence interval

sns.lineplot(x='date', y='value', data=df)

sns.lineplot(x='date', y='value', data=df, hue='category')

  

# Multiple lines

sns.lineplot(data=df, dashes=False, markers=True)

#### Multi-plot Grids

python

# FacetGrid

g = sns.FacetGrid(df, col='category', hue='subcategory')

g.map(sns.scatterplot, 'x', 'y')

g.add_legend()

  

# PairGrid (customized pairplot)

g = sns.PairGrid(df, vars=['col1', 'col2', 'col3'], hue='category')

g.map_diag(sns.histplot)

g.map_offdiag(sns.scatterplot)

  

# JointGrid (combine different plots)

g = sns.JointGrid(data=df, x='x', y='y')

g.plot_joint(sns.scatterplot)

g.plot_marginals(sns.histplot)

  

# Joint plot (quick version)

sns.jointplot(x='x', y='y', data=df, kind='scatter')

sns.jointplot(x='x', y='y', data=df, kind='hex')

sns.jointplot(x='x', y='y', data=df, kind='reg')

sns.jointplot(x='x', y='y', data=df, kind='kde')

#### Additional Statistical Plots

python

# ECDF plot (empirical cumulative distribution)

sns.ecdfplot(data)

  

# Ridge plot (KDE on multiple variables)

sns.ridgeplot(data)

  

# Correlation matrix plot (upper/lower triangle)

corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')

  

# Violin with box plot

sns.violinplot(x='category', y='value', data=df, inner='box')

  

# Scatter with regression line

sns.regplot(x='x', y='y', data=df, ci=95, scatter_kws={'alpha': 0.5})

### Common Mistakes with Seaborn Imports

1. Not importing Matplotlib

python

# Wrong - Seaborn needs Matplotlib

import seaborn as sns

sns.histplot(data)  # May not display

  

# Correct - import and use with Matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

sns.histplot(data)

plt.show()

2. Using Seaborn with non-DataFrame data incorrectly

python

# Wrong - passing Python lists with different lengths

sns.boxplot(x=category_list, y=value_list)  # Error if lengths differ

  

# Correct - use DataFrame

df = pd.DataFrame({'category': category_list, 'value': value_list})

sns.boxplot(x='category', y='value', data=df)

3. Not accounting for Seaborn's default style changes

python

# Seaborn changes Matplotlib defaults

import seaborn as sns

sns.set_style('darkgrid')

# All subsequent Matplotlib plots will use Seaborn style

plt.plot(x, y)  # Uses Seaborn style

  

# Reset to Matplotlib defaults

sns.reset_defaults()

Interview Note: Seaborn is commonly used in data science interviews for EDA. Know how to create common plots (pairplot, heatmap, boxplot, violinplot) and customize them.

Best Practice: Use Seaborn for exploratory analysis and statistical plots. For publications, start with Seaborn and then fine-tune with Matplotlib. Use sns.set_style() and sns.set_palette() to maintain consistent styling.

---

## 3.10 Plotly Imports

### Core Import

python

import plotly.express as px

import plotly.graph_objects as go

import plotly.io as pio

What it imports: Plotly's main modules for creating interactive visualizations.

Why it's used: Plotly provides interactive, web-based visualizations with features like:

- Zoom, pan, and hover tooltips
    
- Dynamic updates and animations
    
- Dashboard creation
    
- 3D visualizations
    
- Integration with Jupyter notebooks and web applications
    

When to use:

- When interactivity is important
    
- For web-based dashboards and applications
    
- For presentations and reports
    
- For 3D and animated visualizations
    

When to avoid:

- For static publication-quality figures
    
- When you need to share plots in PDF format
    
- When the audience doesn't need interactivity
    
- When performance is critical for large datasets
    

### Essential Sub-Imports

python

# Express (high-level interface)

import plotly.express as px

  

# Graph Objects (low-level interface)

import plotly.graph_objects as go

  

# IO functions (saving, displaying)

import plotly.io as pio

  

# Offline mode

import plotly.offline as pyo

  

# Subplots

from plotly.subplots import make_subplots

  

# Figure Factory (specialized plots)

import plotly.figure_factory as ff

### Frequently Used Functions and Their Imports

#### Plotly Express (Quick, High-level)

python

import plotly.express as px

import pandas as pd

  

# Scatter plot

fig = px.scatter(df, x='x', y='y', color='category', size='value')

fig.show()

  

# Line plot

fig = px.line(df, x='date', y='value', color='category')

fig.show()

  

# Bar plot

fig = px.bar(df, x='category', y='value', color='group')

fig.show()

  

# Histogram

fig = px.histogram(df, x='value', color='category', nbins=30)

fig.show()

  

# Box plot

fig = px.box(df, x='category', y='value', color='group')

fig.show()

  

# Violin plot

fig = px.violin(df, x='category', y='value', box=True)

fig.show()

  

# Density heatmap

fig = px.density_heatmap(df, x='x', y='y', color_continuous_scale='Viridis')

fig.show()

  

# 3D scatter plot

fig = px.scatter_3d(df, x='x', y='y', z='z', color='category')

fig.show()

  

# Animated plot

fig = px.scatter(df, x='x', y='y', animation_frame='time', color='category')

fig.show()

  

# Treemap

fig = px.treemap(df, path=['category', 'subcategory'], values='value')

fig.show()

  

# Sunburst

fig = px.sunburst(df, path=['category', 'subcategory'], values='value')

fig.show()

  

# Pie chart

fig = px.pie(df, values='value', names='category')

fig.show()

#### Graph Objects (Detailed Control)

python

import plotly.graph_objects as go

  

# Scatter plot

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', name='Data'))

fig.update_layout(title='My Plot', xaxis_title='X', yaxis_title='Y')

fig.show()

  

# Multiple traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=x1, y=y1, name='Series 1'))

fig.add_trace(go.Scatter(x=x2, y=y2, name='Series 2'))

fig.update_layout(title='Multiple Series')

  

# Bar chart

fig = go.Figure(data=[go.Bar(x=categories, y=values)])

fig.update_layout(title='Bar Chart', xaxis_title='Category', yaxis_title='Value')

  

# 3D surface

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

fig.update_layout(title='3D Surface', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

  

# Sankey diagram

fig = go.Figure(data=[go.Sankey(

    node=dict(

        label=['A', 'B', 'C', 'D'],

        color='blue'

    ),

    link=dict(

        source=[0, 0, 1, 2],

        target=[1, 2, 3, 3],

        value=[8, 4, 2, 8]

    )

)])

#### Subplots and Layouts

python

from plotly.subplots import make_subplots

  

# Create subplots

fig = make_subplots(rows=2, cols=2, subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4'))

fig.add_trace(go.Scatter(x=x, y=y), row=1, col=1)

fig.add_trace(go.Bar(x=categories, y=values), row=1, col=2)

fig.add_trace(go.Histogram(x=data), row=2, col=1)

fig.add_trace(go.Box(y=data), row=2, col=2)

fig.update_layout(height=800)

fig.show()

  

# Update layout

fig.update_layout(

    title='Complex Figure',

    showlegend=True,

    legend=dict(x=0.8, y=1),

    template='plotly_dark',

    hovermode='x unified'

)

#### Customization

python

# Update traces

fig.update_traces(

    marker=dict(size=12, color='red', symbol='circle'),

    line=dict(width=2, color='blue')

)

  

# Update axes

fig.update_xaxes(

    title_text='X Axis',

    range=[0, 10],

    tickvals=[0, 2, 4, 6, 8, 10],

    ticktext=['Zero', 'Two', 'Four', 'Six', 'Eight', 'Ten']

)

  

# Add annotations

fig.add_annotation(

    x=5, y=1,

    text='Important Point',

    showarrow=True,

    arrowhead=1

)

  

# Add shapes

fig.add_shape(

    type='line',

    x0=0, y0=0,

    x1=1, y1=1,

    line=dict(color='red', width=2)

)

#### Saving and Exporting

python

# Save as HTML

fig.write_html('plot.html')

  

# Save as static image

fig.write_image('plot.png', width=800, height=600)

fig.write_image('plot.pdf', width=800, height=600)

fig.write_image('plot.svg', width=800, height=600)

  

# Convert to JSON

json_str = fig.to_json()

fig.from_json(json_str)

  

# Display in Jupyter

pio.show(fig)

pio.renderers.default = 'jupyterlab'  # Set renderer

### Common Mistakes with Plotly Imports

1. Not using the right import for the task

python

# Wrong - trying to use express for complex layout

import plotly.express as px

fig = px.scatter(df, x='x', y='y')

fig.add_trace(go.Bar(...))  # May not work as expected

  

# Better - use graph_objects for mixed plots

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(...))

fig.add_trace(go.Bar(...))

2. Forgetting to show plots

python

# Wrong - no display

fig = px.scatter(df, x='x', y='y')

# Nothing appears

  

# Correct

fig = px.scatter(df, x='x', y='y')

fig.show()

3. Not handling large datasets

python

# Wrong - slows down browser

fig = px.scatter(large_df, x='x', y='y')  # Too many points

  

# Better - sample or aggregate

sampled_df = large_df.sample(1000)

fig = px.scatter(sampled_df, x='x', y='y')

Interview Note: Plotly is increasingly popular in data science interviews for creating interactive dashboards and exploratory visualizations.

Best Practice: Use Plotly Express for quick, interactive visualizations and Graph Objects for complex, custom figures. Always consider the audience: interactive plots work well for presentations, but static plots may be better for documents.

---

## 3.11 Scikit-learn Imports

### Core Import

python

import sklearn

What it imports: The Scikit-learn library for machine learning in Python.

Why it's used: Scikit-learn provides a consistent, easy-to-use interface for machine learning algorithms. It offers:

- Unified API for different algorithms
    
- Comprehensive preprocessing tools
    
- Model evaluation and selection
    
- Pipeline construction
    
- Integration with NumPy and Pandas
    

When to use:

- For machine learning projects
    
- When you need standard algorithms (not deep learning)
    
- For reproducible model training and evaluation
    
- For feature engineering and preprocessing
    

When to avoid:

- For deep learning (use PyTorch or TensorFlow)
    
- For statistical inference (use Statsmodels)
    
- For specialized algorithms not in Scikit-learn
    

### Essential Sub-Imports

python

# Model selection

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

  

# Preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import PolynomialFeatures

  

# Metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from sklearn.metrics import classification_report, mean_squared_error, r2_score

  

# Linear models

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.linear_model import Ridge, Lasso, ElasticNet

  

# Tree-based models

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

  

# Support Vector Machines

from sklearn.svm import SVC, SVR

  

# Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

  

# Naive Bayes

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

  

# Clustering

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

  

# Dimensionality reduction

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

  

# Feature selection

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_selection import RFE, SelectFromModel

  

# Pipelines

from sklearn.pipeline import Pipeline, make_pipeline

### Frequently Used Functions and Their Imports

#### Model Selection and Data Splitting

python

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

  

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)

  

# Cross-validation

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

  

# Grid search

param_grid = {

    'n_estimators': [100, 200, 300],

    'max_depth': [5, 10, None]

}

grid_search = GridSearchCV(

    RandomForestClassifier(),

    param_grid,

    cv=5,

    scoring='accuracy',

    n_jobs=-1

)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")

print(f"Best score: {grid_search.best_score_:.3f}")

  

# Randomized search

param_dist = {

    'n_estimators': [100, 200, 300, 400],

    'max_depth': [5, 10, 15, None]

}

random_search = RandomizedSearchCV(

    RandomForestClassifier(),

    param_distributions=param_dist,

    n_iter=20,

    cv=5,

    scoring='accuracy',

    n_jobs=-1

)

random_search.fit(X_train, y_train)

#### Preprocessing

python

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import PolynomialFeatures

from sklearn.compose import ColumnTransformer

  

# Scaling

scaler = StandardScaler()  # Mean 0, Std 1

X_scaled = scaler.fit_transform(X)

  

minmax_scaler = MinMaxScaler()  # Scale to [0, 1]

X_scaled = minmax_scaler.fit_transform(X)

  

# Encoding

# Label encoding (for target variables)

le = LabelEncoder()

y_encoded = le.fit_transform(y)

  

# One-hot encoding (for categorical features)

encoder = OneHotEncoder()

X_encoded = encoder.fit_transform(X)

  

# Column transformer

preprocessor = ColumnTransformer(

    transformers=[

        ('num', StandardScaler(), ['age', 'salary']),

        ('cat', OneHotEncoder(), ['gender', 'category'])

    ]

)

X_transformed = preprocessor.fit_transform(X)

  

# Polynomial features

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

#### Linear Models

python

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.linear_model import Ridge, Lasso, ElasticNet

  

# Linear Regression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

  

# Ridge (L2 regularization)

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

  

# Lasso (L1 regularization)

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

  

# ElasticNet (L1 + L2)

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic.fit(X_train, y_train)

  

# Logistic Regression

model = LogisticRegression(C=1.0, max_iter=1000)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

y_pred = model.predict(X_test)

#### Tree-based Models

python

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

  

# Decision Tree

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10)

tree.fit(X_train, y_train)

  

# Random Forest

rf = RandomForestClassifier(

    n_estimators=100,

    max_depth=10,

    min_samples_split=5,

    random_state=42,

    n_jobs=-1

)

rf.fit(X_train, y_train)

  

# Feature importance

importances = rf.feature_importances_

  

# Gradient Boosting

gb = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42

)

gb.fit(X_train, y_train)

  

# AdaBoost

ada = AdaBoostClassifier(

    n_estimators=100,

    learning_rate=1.0

)

ada.fit(X_train, y_train)

#### Support Vector Machines

python

from sklearn.svm import SVC, SVR

  

# Classification

svm = SVC(

    kernel='rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'

    C=1.0,

    gamma='scale',

    probability=True

)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

y_proba = svm.predict_proba(X_test)

  

# Regression

svr = SVR(kernel='rbf', C=1.0, gamma='scale')

svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

#### Nearest Neighbors and Naive Bayes

python

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

  

# KNN Classification

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

  

# KNN Regression

knn_reg = KNeighborsRegressor(n_neighbors=5)

knn_reg.fit(X_train, y_train)

y_pred = knn_reg.predict(X_test)

  

# Naive Bayes

gnb = GaussianNB()  # For continuous features

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

  

mnb = MultinomialNB()  # For discrete features (counts)

mnb.fit(X_train, y_train)

#### Clustering

python

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.metrics import silhouette_score

  

# K-Means

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

kmeans.fit(X)

labels = kmeans.labels_

centers = kmeans.cluster_centers_

  

# Find optimal K using elbow method

inertias = []

for k in range(1, 11):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X)

    inertias.append(kmeans.inertia_)

  

# DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)

labels = dbscan.fit_predict(X)  # -1 indicates noise

  

# Agglomerative Clustering

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')

labels = agg.fit_predict(X)

#### Dimensionality Reduction

python

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

  

# PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2f}")

  

# t-SNE (for visualization)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)

X_tsne = tsne.fit_transform(X)

  

# LDA (supervised)

lda = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X, y)

#### Pipelines

python

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

  

# Simple pipeline

pipeline = Pipeline([

    ('scaler', StandardScaler()),

    ('model', RandomForestClassifier())

])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

  

# Make pipeline (shortcut)

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())

  

# Complex pipeline with imputation

numeric_transformer = Pipeline([

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])

  

categorical_transformer = Pipeline([

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

  

preprocessor = ColumnTransformer([

    ('num', numeric_transformer, numeric_features),

    ('cat', categorical_transformer, categorical_features)

])

  

pipeline = Pipeline([

    ('preprocessor', preprocessor),

    ('classifier', RandomForestClassifier())

])

  

pipeline.fit(X_train, y_train)

### Common Mistakes with Scikit-learn Imports

1. Not separating features and target

python

# Wrong

X = df[['feature1', 'feature2']]  # Features

y = df  # Wrong - whole DataFrame

  

# Correct

X = df[['feature1', 'feature2']]

y = df['target']

2. Forgetting to scale before PCA

python

# Wrong - scaling needed for PCA

pca = PCA()

X_pca = pca.fit_transform(X)  # May be biased

  

# Correct - scale first

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

3. Using the same data for training and validation

python

# Wrong - no validation set

model = RandomForestClassifier()

model.fit(X_train, y_train)

score = model.score(X_train, y_train)  # Overly optimistic

  

# Correct - use test set

model = RandomForestClassifier()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)  # Proper evaluation

Interview Note: Scikit-learn is essential for machine learning interviews. Be prepared to explain algorithms, hyperparameters, and model evaluation techniques.

Best Practice: Always scale features for distance-based algorithms. Use pipelines for reproducible workflows. Use cross-validation for reliable model evaluation. Always set random_state for reproducibility.

---

## 3.12 SQLAlchemy Imports

### Core Import

python

from sqlalchemy import create_engine

import sqlalchemy as sa

What it imports: The SQLAlchemy library for SQL database interaction in Python.

Why it's used: SQLAlchemy provides:

- SQL toolkit and Object-Relational Mapping (ORM)
    
- Connection pooling and database abstraction
    
- Parameterized queries
    
- Integration with Pandas for reading/writing data
    
- Support for multiple database backends
    

When to use:

- When connecting to SQL databases
    
- For data extraction and loading
    
- When you need to parameterize SQL queries
    
- For complex database operations
    

When to avoid:

- For simple, one-time queries (use DuckDB or Pandas)
    
- When you don't need database connections
    
- For small datasets that fit in memory
    

### Essential Sub-Imports

python

# Main ORM functionality

from sqlalchemy import create_engine, MetaData, Table, Column

from sqlalchemy import Integer, String, Float, DateTime, Boolean

from sqlalchemy import select, insert, update, delete

from sqlalchemy import text

  

# ORM session management

from sqlalchemy.orm import sessionmaker, Session

from sqlalchemy.ext.declarative import declarative_base

  

# Utility functions

from sqlalchemy import func, desc, asc

from sqlalchemy import and_, or_, not_

  

# Engine and connection

from sqlalchemy.engine import create_engine

from sqlalchemy import event

### Frequently Used Functions and Their Imports

#### Creating Engine and Connecting

python

from sqlalchemy import create_engine

import pandas as pd

  

# SQLite

engine = create_engine('sqlite:///database.db')

engine = create_engine('sqlite:///:memory:')  # In-memory

  

# PostgreSQL

engine = create_engine('postgresql://user:password@localhost:5432/dbname')

engine = create_engine('postgresql+psycopg2://user:password@localhost/dbname')

  

# MySQL

engine = create_engine('mysql+pymysql://user:password@localhost/dbname')

  

# SQL Server

engine = create_engine('mssql+pyodbc://user:password@host/dbname')

  

# Connection options

engine = create_engine(

    'postgresql://user:password@localhost/dbname',

    pool_size=10,

    max_overflow=20,

    pool_timeout=30,

    echo=False

)

  

# Test connection

try:

    with engine.connect() as conn:

        result = conn.execute(text("SELECT 1"))

        print("Connected successfully")

except Exception as e:

    print(f"Connection failed: {e}")

#### Reading and Writing Data

python

import pandas as pd

  

# Read SQL query into DataFrame

df = pd.read_sql("SELECT * FROM table", engine)

df = pd.read_sql("SELECT * FROM table WHERE id > %(id)s", engine, params={'id': 100})

  

# Write DataFrame to SQL table

df.to_sql('table_name', engine, 

          if_exists='replace',  # 'replace', 'append', 'fail'

          index=False,

          chunksize=1000)  # Process in chunks

  

# Read SQL with SQLAlchemy text()

from sqlalchemy import text

query = text("SELECT * FROM table WHERE id > :id")

df = pd.read_sql(query, engine, params={'id': 100})

#### Executing Queries

python

from sqlalchemy import text

  

# Execute raw SQL

with engine.connect() as conn:

    result = conn.execute(text("SELECT * FROM table"))

    for row in result:

        print(row)

  

# Parameterized queries

with engine.connect() as conn:

    query = text("SELECT * FROM table WHERE age > :age AND city = :city")

    result = conn.execute(query, {'age': 25, 'city': 'New York'})

    rows = result.fetchall()

  

# Multiple statements

with engine.connect() as conn:

    conn.execute(text("""

        CREATE TABLE users (

            id INTEGER PRIMARY KEY,

            name VARCHAR(50),

            age INTEGER

        )

    """))

    conn.commit()

#### ORM Basics

python

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import Column, Integer, String, Float, DateTime

from sqlalchemy.orm import sessionmaker

  

# Define model

Base = declarative_base()

  

class User(Base):

    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)

    name = Column(String(50))

    age = Column(Integer)

    email = Column(String(100))

  

# Create tables

Base.metadata.create_all(engine)

  

# Create session

Session = sessionmaker(bind=engine)

session = Session()

  

# Insert data

user = User(name='Alice', age=25, email='alice@example.com')

session.add(user)

session.add_all([

    User(name='Bob', age=30, email='bob@example.com'),

    User(name='Charlie', age=35, email='charlie@example.com')

])

session.commit()

  

# Query data

users = session.query(User).filter(User.age > 25).all()

users = session.query(User).filter(User.name.like('%Bob%')).all()

users = session.query(User).order_by(User.age.desc()).limit(10).all()

  

# Update data

user = session.query(User).filter(User.id == 1).first()

user.age = 26

session.commit()

  

# Delete data

user = session.query(User).filter(User.id == 1).first()

session.delete(user)

session.commit()

#### Advanced Queries

python

from sqlalchemy import func, desc, and_, or_, not_

  

# Aggregate functions

total = session.query(func.count()).select_from(User).scalar()

avg_age = session.query(func.avg(User.age)).scalar()

max_age = session.query(func.max(User.age)).scalar()

  

# Group by

result = session.query(

    User.age, func.count(User.id)

).group_by(User.age).all()

  

# Joins

session.query(User, Profile).join(Profile, User.id == Profile.user_id).all()

  

# Complex conditions

users = session.query(User).filter(

    and_(

        User.age.between(20, 30),

        User.name.like('%A%')

    )

).all()

  

users = session.query(User).filter(

    or_(

        User.age > 30,

        User.name.like('%B%')

    )

).all()

  

# Use text SQL with ORM

from sqlalchemy import text

users = session.query(User).from_statement(

    text("SELECT * FROM users WHERE age > :age")

).params(age=25).all()

### Common Mistakes with SQLAlchemy Imports

1. Not closing connections

python

# Wrong - no connection management

engine = create_engine('sqlite:///database.db')

conn = engine.connect()

result = conn.execute("SELECT * FROM table")

# Connection stays open

  

# Correct - use context manager

with engine.connect() as conn:

    result = conn.execute("SELECT * FROM table")

# Connection automatically closed

2. Mixing ORM and raw SQL incorrectly

python

# Wrong - inconsistent approach

session.query(User).filter("age > 25")  # Raw SQL in filter

  

# Correct - use ORM expressions

session.query(User).filter(User.age > 25)

  

# Or use raw SQL consistently

conn.execute(text("SELECT * FROM users WHERE age > 25"))

3. Forgetting to commit transactions

python

# Wrong - no commit

user = User(name='Alice')

session.add(user)

# Data not saved

  

# Correct - commit

user = User(name='Alice')

session.add(user)

session.commit()

  

# Or use context manager for automatic commit

with session.begin():

    session.add(user)

Interview Note: SQLAlchemy is often tested in data engineering interviews. Know how to connect to databases, execute queries, and load data into DataFrames.

Best Practice: Use context managers for connections and transactions. Use if_exists='replace' with caution. For large data writes, use chunksize to avoid memory issues.

---

## 3.13 PyArrow Imports

### Core Import

python

import pyarrow as pa

import pyarrow.parquet as pq

import pyarrow.csv as csv

import pyarrow.json as json

import pyarrow.ipc as ipc  # For Feather format

What it imports: The PyArrow library, which provides Python bindings for Apache Arrow.

Why it's used: Apache Arrow provides a columnar memory format that enables efficient data processing and exchange. PyArrow offers:

- Zero-copy data sharing with Pandas, Polars, etc.
    
- Efficient serialization/deserialization
    
- Parquet and Feather file support
    
- In-memory columnar processing
    
- Integration with other Arrow-compatible systems
    

When to use:

- When performance and memory efficiency matter
    
- For data exchange between different systems
    
- For reading/writing Parquet, Feather, or Arrow formats
    
- For large-scale data processing
    

When to avoid:

- For small datasets (overhead not worth it)
    
- When you don't need the Arrow ecosystem
    
- For development work that doesn't involve large data
    

### Essential Sub-Imports

python

# Core Arrow library

import pyarrow as pa

  

# Parquet support

import pyarrow.parquet as pq

  

# CSV support

import pyarrow.csv as csv

  

# JSON support

import pyarrow.json as json

  

# IPC/Feather support

import pyarrow.ipc as ipc

  

# Filesystem support

import pyarrow.fs as fs

  

# Compute functions

import pyarrow.compute as pc

  

# Dataset support (for large data)

import pyarrow.dataset as ds

  

# Date/time utilities

import pyarrow.compute as pc

### Frequently Used Functions and Their Imports

#### Creating Arrow Arrays and Tables

python

import pyarrow as pa

import numpy as np

  

# Create arrays

arr = pa.array([1, 2, 3, 4, 5])

arr = pa.array([1, 2, None, 4, 5])  # With nulls

arr = pa.array(np.array([1, 2, 3, 4, 5]))

arr = pa.array(np.random.randn(1000))

  

# Create from pandas

import pandas as pd

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

table = pa.Table.from_pandas(df)

  

# Create from dictionaries

table = pa.Table.from_pydict({

    'col1': [1, 2, 3],

    'col2': ['a', 'b', 'c']

})

  

# Create from lists of arrays

table = pa.Table.from_arrays(

    [pa.array([1, 2, 3]), pa.array(['a', 'b', 'c'])],

    names=['col1', 'col2']

)

  

# Create table with schema

schema = pa.schema([

    ('col1', pa.int64()),

    ('col2', pa.string())

])

table = pa.Table.from_pydict({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}, schema=schema)

#### Working with Parquet

python

import pyarrow.parquet as pq

  

# Write to Parquet

pq.write_table(table, 'data.parquet')

  

# Read Parquet file

table = pq.read_table('data.parquet')

df = table.to_pandas()

  

# Parquet with options

pq.write_table(

    table,

    'data.parquet',

    compression='snappy',  # 'snappy', 'gzip', 'lz4', 'zstd'

    row_group_size=10000,  # Number of rows per row group

    data_page_size=1048576  # 1MB data pages

)

  

# Read Parquet with filtering

table = pq.read_table(

    'data.parquet',

    columns=['col1', 'col2'],  # Only read specific columns

    filter=('col1', '=', 5)    # Row filtering

)

  

# Read Parquet metadata

metadata = pq.read_metadata('data.parquet')

print(f"Number of rows: {metadata.num_rows}")

print(f"Number of row groups: {metadata.num_row_groups}")

print(f"Schema: {metadata.schema}")

#### Working with CSV and JSON

python

import pyarrow.csv as csv

import pyarrow.json as json

  

# Read CSV

table = csv.read_csv('data.csv')

table = csv.read_csv('data.csv', 

                     skip_rows=1,  # Skip header

                     parse_options=csv.ParseOptions(delimiter=',', quote_char='"'))

  

# Write CSV

csv.write_csv(table, 'output.csv')

  

# Read JSON

table = json.read_json('data.json')

table = json.read_json('data.json', 

                       parse_options=json.ParseOptions(newlines_in_values=True))

#### Working with Feather/IPC

python

import pyarrow.ipc as ipc

  

# Write Feather file

ipc.write_feather(table, 'data.feather')

  

# Read Feather file

table = ipc.read_feather('data.feather')

  

# Streaming IPC

with ipc.new_stream('data.stream', table.schema) as writer:

    writer.write_table(table)

  

with ipc.open_stream('data.stream') as reader:

    table = reader.read_all()

#### Data Manipulation with Arrow

python

import pyarrow.compute as pc

  

# Filtering

table = pa.table({

    'col1': [1, 2, 3, 4, 5],

    'col2': ['a', 'b', 'c', 'd', 'e']

})

filtered = table.filter(pc.field('col1') > 3)

  

# Column operations

table = table.append_column('col3', pc.multiply(pc.field('col1'), 2))

table = table.set_column(2, 'col3', pc.add(pc.field('col1'), pc.field('col2')))

  

# Aggregations

sum_col1 = pc.sum(pc.field('col1'))

mean_col1 = pc.mean(pc.field('col1'))

  

# String operations

table = table.append_column('upper_col2', pc.utf8_upper(pc.field('col2')))

  

# Date operations

table = table.append_column('year', pc.year(pc.field('date_col')))

#### Dataset Support (Large Data)

python

import pyarrow.dataset as ds

  

# Read dataset (supports multiple files)

dataset = ds.dataset('data/*.parquet', format='parquet')

  

# Query dataset

scanner = dataset.scanner(

    filter=ds.field('col1') > 10,

    columns=['col1', 'col2'],

    batch_size=10000

)

table = scanner.to_table()

  

# Write dataset

ds.write_dataset(

    table,

    'output/',

    format='parquet',

    partitioning=['year', 'month']

)

### Common Mistakes with PyArrow Imports

1. Not handling null values correctly

python

# Wrong - pandas style

arr = pa.array([1, 2, None, 4])

arr.fillna(0)  # Doesn't exist

  

# Correct - use compute functions

arr = pc.coalesce(arr, pa.scalar(0))

2. Forgetting to convert back to pandas

python

# Wrong - Arrow table not directly usable with pandas functions

table = pq.read_table('data.parquet')

print(table.head())  # Not pandas DataFrame

  

# Correct - convert when needed

df = table.to_pandas()

print(df.head())

3. Not using the right compression

python

# Wrong - default compression may not be optimal

pq.write_table(table, 'data.parquet')  # Default compression

  

# Better - choose compression based on use case

pq.write_table(table, 'data.parquet', compression='zstd', compression_level=9)

Interview Note: PyArrow is becoming more important in data engineering interviews. Know how it enables efficient data transfer and storage.

Best Practice: Use PyArrow for data exchange, especially when working with Parquet files. Use to_pandas() when you need to work with pandas, but keep data in Arrow format when possible for performance.

---

## 3.14 Utility Imports

### Standard Library Utilities

python

# System and path operations

import os

import sys

from pathlib import Path

  

# File operations

import json

import pickle

import csv

import gzip

import zipfile

import shutil

import tempfile

  

# Date and time

from datetime import datetime, date, timedelta

import time

import calendar

  

# Math and random

import math

import random

from random import randint, choice, sample

import itertools

import collections

  

# Regular expressions

import re

  

# Parallel processing

import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

  

# Logging

import logging

  

# Warnings

import warnings

warnings.filterwarnings('ignore')  # Suppress warnings

  

# Type hints

from typing import List, Dict, Tuple, Optional, Union, Any

### Third-party Utilities

python

# Data science tools

import numpy as np

import pandas as pd

  

# Jupyter notebook-specific

from IPython.display import display, HTML, Markdown

from tqdm import tqdm, tqdm_notebook  # Progress bars

  

# Configuration and environment

import yaml  # YAML file support

import dotenv  # Environment variables

from dotenv import load_dotenv

  

# Data validation

import pandas as pd

from pandas.api.types import is_numeric_dtype, is_string_dtype

  

# Web scraping

import requests

from urllib.parse import urljoin, urlparse

### Examples of Utility Usage

python

# Path handling

from pathlib import Path

data_dir = Path('data')

data_dir.mkdir(parents=True, exist_ok=True)

csv_path = data_dir / 'data.csv'

  

# JSON handling

import json

data = json.load(open('config.json'))

with open('output.json', 'w') as f:

    json.dump(data, f, indent=2)

  

# Pickle handling

with open('model.pkl', 'wb') as f:

    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:

    model = pickle.load(f)

  

# Date operations

from datetime import datetime, timedelta

now = datetime.now()

yesterday = now - timedelta(days=1)

date_str = now.strftime('%Y-%m-%d')

  

# Progress bars

from tqdm import tqdm

for i in tqdm(range(1000)):

    process_item(i)

  

# Parallel processing

from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:

    results = list(executor.map(process_func, data_list))

---

## 3.15 Import Best Practices

### 1. Standard Import Ordering

python

# 1. Standard library imports

import os

import sys

from datetime import datetime

import re

  

# 2. Third-party library imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

  

# 3. Local application imports

from mymodule import function

### 2. Wildcard Imports

python

# Avoid this

from sklearn import *  # Bad practice

  

# Use explicit imports

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

### 3. Import Aliases

python

# Standard aliases (community accepted)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

  

# Avoid custom aliases unless necessary

# Bad: import numpy as nmp

# Good: import numpy as np

### 4. Lazy Imports for Performance

python

def create_plot():

    import matplotlib.pyplot as plt  # Import only when needed

    plt.plot([1, 2, 3])

    plt.show()

### 5. Managing Dependencies

python

# requirements.txt

numpy>=1.21.0

pandas>=1.3.0

matplotlib>=3.4.0

  

# Or use environment.yml for conda

# environment.yml

name: my_env

dependencies:

  - python=3.9

  - numpy=1.21.0

  - pandas=1.3.0

### 6. Import for Development vs Production

python

# Development only imports

try:

    from IPython.display import display

except ImportError:

    pass

  

# Production imports

python

```
import logging
```

python

```
import warnings
```

python

```
warnings.filterwarnings('ignore')
```
### Common Import Mistakes to Avoid

1. Circular imports: Module A imports from B, and B imports from A
    
2. Importing everything: Using from module import *
    
3. Catching ImportError incorrectly: Using bare except with imports
    
4. Not checking Python version: Some features require specific Python versions
    

### Import Performance Considerations

python

```
import time

start = time.time()

import pandas as pd

print(f"Import time: {time.time() - start:.2f}s")

  

# Reduce import time by importing only what's needed

from pandas import read_csv, DataFrame  # Faster than importing all
```
---

This comprehensive import reference covers all major libraries in the Python data analysis ecosystem. Each section provides the essential imports, common functions, and best practices for real-world data science work. Remember that proper imports are the foundation of clean, efficient, and maintainable data analysis code.

  
**