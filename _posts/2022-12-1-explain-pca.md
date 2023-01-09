---
layout: post
title: "Explain PCA Like I'm 5"
description: A simple walkthrough of why PCA is needed and how to interpret it.
date: 2023-1-9
tags: tutorial data-science
giscus_comments: true
---

Hi! You are either:

- not a scientist but want to understand whether PCA should be used
- a data scientist but want to explain to a colleague why PCA should be used

I recently realised that explaining Principal Component Analysis (PCA) to non-scientists is probably a universal experience all data scientists eventually need to do.

I will explain the cases where PCA is needed, how PCA works, and how to interpret the new features produced by PCA.

If you have further questions not addressed in the FAQ, please leave a comment and I will answer you.

<hr>

## Why is PCA needed?

PCA reduces the number of dimensions of the dataset.

Thus, PCA is needed if you:

- have a very high-dimensional dataset (e.g. ≥100 features)
- cannot afford a large amount of compute (minutes vs hours vs days)
- want to make sense of many features (e.g. to select important features)

<hr>

## What is PCA doing?

In a sentence:

<blockquote>
    PCA produces new features as a <u><b>weighted sum</b></u> of original features, by <u><b>preserving variance</b></u> in the original features.
</blockquote>

What is a <b>weighted sum<b>?

$$ x_\text{new feature} = w_1\cdot x_\text{feature 1} + w_2\cdot x_\text{feature 2} + ... + w_n\cdot x_\text{feature n} $$

$$x_\text{new feature}$$ is a sum of original features $$x_\text{feature 1}, ..., x_\text{feature n}$$, each weighted by $$w_1,...,w_n$$ respectively.

Why preserve variance?

Preserving variance in the original features retains information about the spread of data, usually resulting in good representations of original features.

<hr>

## Example: PCA on Housing Data

Let’s walk through PCA with an example:

- Given a dataset of information about houses, you want to predict its sales price. However, dealing with so many features is unwieldy and expensive in compute.
    - Input values: 38 numerical features
    - Output values: Sale price

<details>
<summary>A sample of the housing dataset:</summary>
{% highlight python %}
import pandas as pd

!kaggle competitions download -c house-prices-advanced-regression-techniques
!unzip house-prices-advanced-regression-techniques.zip

train = pd.read_csv('train.csv')
df = train._get_numeric_data().dropna()
df.head()
{% endhighlight %}
</details>

<table border="1" class="dataframe" style="width: 100%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>...</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>...</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>...</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>...</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>...</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>...</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
5 rows x 38 columns

Because PCA reduces the number of dimensions of the dataset, model training takes a shorter time without sacrificing performance.

<details>
<summary>Running PCA on original features:</summary>
{% highlight python %}
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df.drop(columns='SalePrice').values
y = np.log(df['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=2)
pca.fit(X_train)
X_features = pca.transform(X_train)
X_features_test = pca.transform(X_test)

print("Original Features:\n", X_train[0].round(3))
print("\nPCA Features:\n", X_features[0].round(3))
{% endhighlight %}
</details>
```
Original Features:
 [-0.646  0.136  0.171 -0.252  1.297 -0.523  1.221  1.129  0.026  0.221
 -0.283 -0.675 -0.531 -0.856  1.734 -0.106  0.784  1.175 -0.233  0.79
  1.238  0.175 -0.202  0.9   -0.943  1.214  0.211  0.271 -0.746  1.613
 -0.363 -0.114 -0.285 -0.074 -0.146  2.113  0.879]

PCA Features:
 [ 2.141 -1.273]
```

<hr>

### Performance

Training 2 linear regression models, optimizing for root mean squared error (RMSE) of the logarithm of sale price (as evaluated on Kaggle), even though PCA reduces dimensions from 37 to merely 2, there is only a small decrease in test error of -0.02.

<details>
<summary>Comparing original features and PCA features:</summary>
{% highlight python %}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_hat = lr.predict(X_train)
y_hat = lr.predict(X_test)

print("Original Features (37 dimensions)")
print("Train error:", np.sqrt(mean_squared_error(y_train, y_train_hat)).round(4)) 
print("Test error:", np.sqrt(mean_squared_error(y_test, y_hat)).round(4))

lr_pca = LinearRegression()
lr_pca.fit(X_features, y_train)
y_train_hat = lr_pca.predict(X_features)
y_hat = lr_pca.predict(X_features_test)

print("\nPCA features (2 dimensions)")
print("Train error:", np.sqrt(mean_squared_error(y_train, y_train_hat)).round(4)) 
print("Test error:", np.sqrt(mean_squared_error(y_test, y_hat)).round(4))
{% endhighlight %}
</details>

```
Original Features (37 dimensions)
Train error: 0.15
Test error: 0.1348

PCA features (2 dimensions)
Train error: 0.1935
Test error: 0.1562
```

<hr>

### Time

With PCA, because number of features is drastically reduced (from 37 to 2), time is also reduced from 4.85ms to 584µs, a 8300X speedup!

<details>
<summary>Training a linear regression model with original features:</summary>
{% highlight python %}
%%timeit

lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_hat = lr.predict(X_train)
y_hat = lr.predict(X_test)
{% endhighlight %}
</details>

```
4.85 ms ± 276 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

<details>
<summary>Training a linear regression model with PCA features:</summary>
{% highlight python %}
%%timeit

lr_pca = LinearRegression()
lr_pca.fit(X_features, y_train)
y_train_hat = lr_pca.predict(X_features)
y_hat = lr_pca.predict(X_features_test)
{% endhighlight %}
</details>
```
584 µs ± 19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

<hr>

### Space

Because the no. of dimensions decreased by 18.5X, so did space taken, from 265Kb to 14Kb.

<details>
<summary>Code</summary>
{% highlight python %}
print(f"Memory size of data w/o PCA: {X_train.size * X_train.itemsize} bytes")
print(f"Memory size of data w/ PCA: {X_features.size * X_features.itemsize} bytes")
{% endhighlight %}
</details>
```
Memory size of data w/o PCA: 265216 bytes
Memory size of data w/ PCA: 14336 bytes
```

<hr>

## How to interpret PCA results

Recall what PCA does:

<blockquote>PCA produces new features as a <u><b>weighted sum</b></u> of original features, by <u><b>preserving variance</b></u> in the original features.</blockquote>

This means that PCA features represent original features as a weighted sum.

$$
\begin{align}
\begin{bmatrix}X_1\\X_2\end{bmatrix}
=&
\begin{bmatrix}PC_1\\PC_2\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\\
=&
\begin{bmatrix}
w_{11} & w_{12} & w_{13}\\
w_{21} & w_{22} & w_{23}
\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\\
=&
\begin{bmatrix}
w_{11}x_1 + w_{12}x_2 + w_{13}x_3\\
w_{21}x_1 + w_{22}x_2 + w_{23}x_3
\end{bmatrix}
\end{align}
$$

In other words, we can get $$X_1$$ and $$X_2$$ using $$PC_1$$ and $$PC_2$$, by computing a weighted sum of $$[x_1, x_2, x_3]$$.

$$ X_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 $$

$$ X_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 $$

<details>
<summary>Compare weighted sums with PCA features:</summary>
{% highlight python %}
print("PC1 weighted sum:", sum(pca.components_[0] * X_train[0]).round(3))
print("PC2 weighted sum:", sum(pca.components_[1] * X_train[0]).round(3))
print("PCA features:", X_features[0].round(3))
{% endhighlight %}
</details>

```
PC1 weighted sum: 1.515
PC2 weighted sum: 0.968
PCA features: [1.515 0.968]
```

Inspecting principal component 1, the highest weighted feature is `OverallQual`. This means that `OverallQual` has the highest variance of all features in the dataset, and is likely a useful feature for sale price prediction.

<details>
<summary>Sort features by the 1st principal component's weights:</summary>
{% highlight python %}
import matplotlib.pyplot as plt
plt.style.use('seaborn')

i = np.abs(pca.components_[0]).argsort()
pca_component_1 = pca.components_[0][i]
columns = df.drop(columns='SalePrice').columns[i]

plt.barh(columns, pca_component_1)
plt.show()
{% endhighlight %}
</details>

{% include figure.html path="assets/img/pca.png" class="img-fluid rounded z-depth-1" %}

## FAQ

<details><summary>
How does PCA preserve variance?</summary>
WIP
</details>

<details><summary>When should I not use PCA?</summary>
WIP
</details>