---
layout: post
title: "PCA Explained Simply"
description: A simple explainer of why PCA is needed and how to interpret it.
date: 2023-1-9
tags: tutorial data-science
giscus_comments: true
---

Hi! If you are reading this, you are either:

- not a scientist but want to understand why PCA should be used
- a data scientist but want to explain to a colleague why PCA should be used

I recently realised that explaining Principal Component Analysis (PCA) to non-scientists is a universally common experience that all data scientists eventually do.

I will explain when PCA is needed, what PCA does, an example with PCA, what benefits PCA brings, and how to interpret the results of PCA.

I will <b>not</b> detail how PCA works, how to use PCA, or whether to use PCA or other methods, but I link nice resources in the FAQ.

If you have questions not addressed in the FAQ, please leave a comment and I will answer you.

<hr><br>


## Why is PCA needed?

PCA reduces the number of dimensions of the dataset.

<div class="row mt-3">
  <div class="col-sm-2 mt-3 mt-md-0"></div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/cat-shadow.jpg" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-2 mt-3 mt-md-0"></div>
</div>

<div class="caption">
  Credits to <a href="https://unsplash.com/photos/JIuZZ0-EC0U">Evan Qu</a>.
</div>

Like how shadows are a 2D representation of 3D objects, PCA finds the best angle to cast the shadow.

Thus, PCA is needed if you:

- have a very high-dimensional dataset (e.g. ≥100 features)
- cannot afford a large amount of compute (minutes vs hours vs days)
- want to make sense of many features (e.g. to select important features)

By applying PCA, you:
- save time (during model training and model iteration)
- save space (in memory and model parameters) 
- trade-off a little bit of performance

<hr><br>

## What does PCA do?

In a sentence:

<blockquote>
    PCA produces new features as a <u><b>weighted sum</b></u> of the original features in the dataset, by <u><b>preserving variance</b></u> in the original features.
</blockquote>

What is a <b>weighted sum</b>?

$$ X = w_1\cdot x_1 + \dotsb + w_n\cdot x_n $$

$$X$$ is a sum of values $$x_1,\dotsc, x_n$$, each weighted by $$w_1,\dotsc, w_n$$ respectively. Likewise, PCA produces new features, e.g. $$X_1, X_2,\dotsc$$, as weighted sums of original features $$x_1,\dotsc, x_n$$.

Why <b>preserve variance</b>?

Preserving variance in the original features retains information about the spread of features in the data, usually resulting in good representations of original features for machine learning.

<hr><br>

## Show me an example!

Let’s walk through PCA with the <a href="https://www.kaggle.com/c/boston-housing">Boston Housing dataset</a>:

Given a dataset of 37 numerical features of houses, you want to predict its sale price. However, dealing with so many features can be (imagine) expensive in compute and difficult to interpret.

<details>
<summary><small>Code</small></summary>
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

Because PCA reduces the number of dimensions of the dataset, model training can take a shorter time without sacrificing too much performance.

Let's run PCA on the 37 features:

<details>
<summary><small>Code</small></summary>
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

print("Original features (normalized):\n", X_train[0].round(3))
print("\nPCA features:\n", X_features[0].round(3))
{% endhighlight %}
</details>
```
Original features (normalized):
 [-0.646  0.136  0.171 -0.252  1.297 -0.523  1.221  1.129  0.026  0.221
 -0.283 -0.675 -0.531 -0.856  1.734 -0.106  0.784  1.175 -0.233  0.79
  1.238  0.175 -0.202  0.9   -0.943  1.214  0.211  0.271 -0.746  1.613
 -0.363 -0.114 -0.285 -0.074 -0.146  2.113  0.879]

PCA features:
 [ 2.141 -1.273]
```

<hr><br>

## What benefits does PCA bring? At what trade-off?

Let's train 2 linear regression models:
1. Trained on the original 37-dimensional features
2. Trained on the 2-dimensional features produced by PCA

<br>

#### Performance: A Tiny Trade-off

Even though PCA reduces 37 dimensions to merely 2 dimensions, there is a small decrease in test error of only <b>-0.02</b> (RMSE of log of sale price as evaluated on Kaggle).

<details>
<summary><small>Code</small></summary>
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

<br>

#### Training Time: A Major Speed-up

With PCA, because number of features is drastically reduced (from 37 to 2), training time taken is also reduced from 4.85ms to 584µs, a <b>8300X</b> speedup!

<details>
<summary><small>Code (Linear regression with original features)</small></summary>
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
<summary><small>Code (Linear regression with PCA features)</small></summary>
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

<br>

#### Memory Space: A Significant Reduction

Because the no. of dimensions decreased by 18.5X, so did memory space used, <b>from 265Kb to 14Kb</b>.

<details>
<summary><small>Code</small></summary>
{% highlight python %}
print(f"Memory size of data w/o PCA: {X_train.size * X_train.itemsize} bytes")
print(f"Memory size of data w/ PCA: {X_features.size * X_features.itemsize} bytes")
{% endhighlight %}
</details>
```
Memory size of data w/o PCA: 265216 bytes
Memory size of data w/ PCA: 14336 bytes
```

<hr><br>

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

Compare weighted sums with PCA features:
<details>
<summary><small>Code</small></summary>
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

Sort features by the 1st principal component's weights:
<details>
<summary><small>Code</small></summary>
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

<hr><br>

## FAQ

<details><summary>
How does PCA preserve variance?</summary>
WIP
</details>

<details><summary>When should I not use PCA?</summary>
WIP
</details>