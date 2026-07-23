# Python Data Analysis Handbook

## Chapter 8: Machine Learning Fundamentals

### Section 8.1 — Linear Regression and Logistic Regression

---

## Table of Contents (Section 8.1)

- [8.1.0 How This Chapter Is Organized](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#810-how-this-chapter-is-organized)
- [8.1 Linear Regression and Logistic Regression](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#81-linear-regression-and-logistic-regression)
    - [8.1.1 Linear Regression](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#811-linear-regression)
    - [8.1.2 Logistic Regression](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#812-logistic-regression)
    - [8.1.3 Linear vs. Logistic Regression: Choosing the Right One](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#813-linear-vs-logistic-regression-choosing-the-right-one)
    - [8.1.4 Common Mistakes](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#814-common-mistakes)
    - [8.1.5 Best Practices](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#815-best-practices)
    - [8.1.6 Interview Notes](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#816-interview-notes)
    - [8.1.7 Section Summary](https://claude.ai/chat/6350c067-4ddc-425a-9912-e63381b591c8#817-section-summary)

> **Prerequisite recap:** Chapter 5 (§5.5) introduced regression and classification as _concepts_ — predicting a number, or predicting a category. This chapter is where those concepts finally become real, working algorithms. Everything here builds on statistics from Chapter 5 (mean, variance, correlation) and clean data from Chapter 6. `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`, and tools from `sklearn` are assumed throughout, as introduced in Chapter 2, §2.11.

---

## 8.1.0 How This Chapter Is Organized

Chapter 8 covers every major machine learning algorithm mentioned back in Chapter 5 (§5.5), split across eight sections so each one gets the depth it deserves:

- **8.1 (this section)** — Linear Regression and Logistic Regression, the two foundational starting points.
- **8.2** — Decision Trees and Random Forest.
- **8.3** — Gradient Boosting and XGBoost.
- **8.4** — Support Vector Machines (SVM) and K-Nearest Neighbors (KNN).
- **8.5** — Naive Bayes.
- **8.6** — Clustering: K-Means, DBSCAN, and Hierarchical Clustering.
- **8.7** — PCA, with the full mathematics behind the concept introduced back in Chapter 5, §5.5.5.
- **8.8** — Putting it all together: how to properly build, train, and evaluate a model without fooling yourself.

Every algorithm in this chapter follows the same structure: what it is, the intuition behind it, the math (explained simply, with a picture wherever the math alone would be hard to follow), its key settings (called **hyperparameters**), its strengths and weaknesses, how to build it in Python, and a real business example.

---

# 8.1 Linear Regression and Logistic Regression

## 8.1.1 Linear Regression

### What Is It?

**Linear regression** is the algorithm version of the straight-line prediction idea first introduced in Chapter 5 (§5.5.2). It finds the single best straight line (or, with more than one input, a flat plane or higher-dimensional equivalent) through your data, and uses that line to predict a number.

### The Simple Way to Think About It

Picture a scatter plot of house size versus house price. Some houses are above the "typical" price for their size, some are below it. Linear regression's whole job is to draw one single straight line through the middle of that cloud of points — positioned so that, overall, it sits as close as possible to every point at once. Once that line exists, you can use it to guess the price of a brand-new house just by knowing its size.

### The Mathematics, Built Up Slowly

**With one input variable**, the line is:

$$\hat{y} = \beta_0 + \beta_1 x$$

Here, $\hat{y}$ (read "y-hat") is the predicted value, $x$ is the input (like house size), $\beta_1$ is the slope (how much the prediction changes for every extra unit of $x$), and $\beta_0$ is the intercept (the predicted value when $x$ is zero). This is the exact same straight-line formula from Chapter 5, §5.5.2 — a model is just a chosen $\beta_0$ and $\beta_1$.

**With several input variables** (like size, number of bedrooms, and age of the house all at once), the idea extends naturally:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \dots$$

Each input now gets its own slope, called a **coefficient**, telling you how much that specific input pushes the prediction up or down, holding everything else steady.

### How Does the Algorithm Actually Pick the Best Line?

This is the part that trips up nearly every beginner, so it's worth slowing down completely.

**Step 1: Define what "best" means.** For any candidate line, you can measure how wrong it is for every single point by looking at the **residual** — the gap between the line's prediction and the real, actual value. Linear regression specifically squares every residual (exactly like the variance formula in Chapter 5, §5.1.7) and adds them all up. This total is called the **cost function** (or **loss function**), and for linear regression it's specifically called **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

In plain words: for every point, find how far off the line's guess was, square that gap (so it's always positive, and big mistakes count extra), and average all those squared gaps together. **A "good" line is simply one with a low MSE.** A "perfect" line, passing exactly through every point, would have an MSE of zero.

**Step 2: Find the line that makes this cost as small as possible.** This is where an algorithm called **gradient descent** comes in — and it's genuinely worth understanding visually, since the word "gradient" scares off a lot of beginners for no good reason.

### Seeing It: Gradient Descent, Visualized

Imagine you are standing somewhere on a big, bowl-shaped hill in thick fog, and your goal is to reach the very bottom of the bowl — but you can't see more than a foot in front of you. A sensible strategy: feel which direction slopes downward right where you're standing, take a small step in that direction, and repeat. Eventually, step by step, you'll reach the bottom. **This is exactly what gradient descent does**, except instead of a physical hill, the "bowl" is the cost function (MSE) plotted against every possible combination of $\beta_0$ and $\beta_1$ — and the "bottom of the bowl" is the specific combination of $\beta_0$ and $\beta_1$ that produces the lowest possible MSE, which is exactly the best-fit line you're looking for.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.linspace(0, 10, 40)
y_true = 3 + 2 * x + np.random.normal(0, 2, 40)

def compute_mse(b0, b1, x, y):
    y_pred = b0 + b1 * x
    return np.mean((y - y_pred) ** 2)

# Build a grid of possible (b0, b1) combinations, and calculate the MSE "bowl" for each one
b0_range = np.linspace(-5, 12, 100)
b1_range = np.linspace(-1, 5, 100)
B0, B1 = np.meshgrid(b0_range, b1_range)
MSE = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        MSE[i, j] = compute_mse(B0[i, j], B1[i, j], x, y_true)

# Simulate gradient descent taking small steps down the bowl
b0, b1 = -4, 4.5   # a deliberately bad starting guess
lr = 0.01
path = [(b0, b1)]
for _ in range(60):
    y_pred = b0 + b1 * x
    grad_b0 = -2 * np.mean(y_true - y_pred)
    grad_b1 = -2 * np.mean((y_true - y_pred) * x)
    b0 -= lr * grad_b0
    b1 -= lr * grad_b1
    path.append((b0, b1))
path = np.array(path)

plt.figure(figsize=(8, 6))
plt.contour(B0, B1, MSE, levels=25, cmap="viridis")
plt.plot(path[:, 0], path[:, 1], color="red", marker="o", markersize=3, linewidth=1.5, label="Gradient descent's path")
plt.scatter([path[-1, 0]], [path[-1, 1]], color="black", s=100, zorder=5, label="Final answer (lowest point in the bowl)")
plt.xlabel("β₀ (intercept)")
plt.ylabel("β₁ (slope)")
plt.title("Gradient Descent: Walking Downhill to the Best-Fit Line")
plt.legend()
plt.colorbar(label="MSE (Cost)")
plt.show()
```

**How to read this picture, in simple words:** 
every point on this chart represents one _possible_ line — one specific combination of intercept ($\beta_0$) and slope ($\beta_1$). The colored contour rings work exactly like a topographic map of hills: rings close together mean a steep slope, and the darkest, innermost ring marks the very bottom of the bowl — the one specific line with the lowest possible MSE. The red dotted path shows gradient descent starting from a deliberately bad, wrong guess (far from the true best answer) and taking small, repeated steps, always downhill, until it settles at the black dot in the middle — the best-fit line. **This is the single most important insight in this whole subsection**: gradient descent isn't some mysterious black box; it is nothing more than "repeatedly nudge your guess a little bit in whichever direction reduces the error," done automatically, many times, until the nudges stop making a meaningful difference.

### A Crucial Setting: The Learning Rate

The size of each "step" gradient descent takes is called the **learning rate**. This is genuinely worth understanding well, because getting it wrong is one of the most common practical mistakes in all of machine learning.

```python
import numpy as np
import matplotlib.pyplot as plt

def run_gradient_descent(lr, steps=40):
    b0, b1 = -4, 4.5
    path = [(b0, b1)]
    for _ in range(steps):
        y_pred = b0 + b1 * x
        grad_b0 = -2 * np.mean(y_true - y_pred)
        grad_b1 = -2 * np.mean((y_true - y_pred) * x)
        b0 -= lr * grad_b0
        b1 -= lr * grad_b1
        path.append((b0, b1))
    return np.array(path)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
learning_rates = [0.001, 0.01, 0.05]
titles = ["Too Small: Painfully Slow", "Just Right: Reaches the Bottom Smoothly", "Too Large: Overshoots and Bounces Around"]

for ax, lr, title in zip(axes, learning_rates, titles):
    path = run_gradient_descent(lr)
    ax.contour(B0, B1, MSE, levels=25, cmap="viridis")
    ax.plot(path[:, 0], path[:, 1], color="red", marker="o", markersize=2, linewidth=1)
    ax.set_title(f"Learning Rate = {lr}\n{title}")
    ax.set_xlabel("β₀")
    ax.set_ylabel("β₁")

plt.tight_layout()
plt.show()
```

**How to read this picture, in simple words:** 
all three panels start from the exact same bad guess, use the exact same data, and run the exact same number of steps — the _only_ thing that changes is the size of each step. In the left panel, the learning rate is so tiny that after 40 steps, the path has barely moved — it would technically get there eventually, but only after an enormous, wasteful number of extra steps. In the middle panel, the steps are sized sensibly, and the path glides smoothly and efficiently down to the bottom of the bowl. In the right panel, the steps are so large that each one overshoots past the bottom of the bowl entirely, causing the path to zig-zag and bounce around chaotically — in a genuinely bad case, a learning rate this large can even bounce further and further away from the answer instead of converging on it. **This is exactly why the learning rate is called a hyperparameter you have to choose carefully** — it's a dial that requires actual thought and testing, not just an automatic setting the algorithm figures out on its own.

### Hyperparameters

|Hyperparameter|What It Controls|Practical Guidance|
|---|---|---|
|**Learning rate**|The size of each gradient descent step|Too small wastes time; too large can overshoot and fail to converge, exactly as shown above|
|**Regularization strength (alpha)**|How strongly the model is discouraged from relying too heavily on any one input (covered next)|Higher values simplify the model more; needs to be tuned, not guessed|
|**Number of iterations**|How many gradient descent steps are allowed before stopping|Too few can stop before reaching the bottom of the bowl; too many wastes computing time once the bottom is already reached|

### Regularization: Ridge and Lasso

**Why This Exists**

Imagine a model with 50 input columns, several of which are genuinely useless or just noisy. Plain linear regression, left unchecked, can end up assigning a large, confident coefficient to a column that's actually just noise — essentially "memorizing" quirks of the specific data it was trained on rather than learning a real, generalizable pattern. This problem, called **overfitting**, is one of the most important ideas in all of machine learning, and it's covered again in full in Section 8.8. **Regularization** is a direct fix: it adds a penalty to the cost function specifically for having large coefficients, gently discouraging the model from leaning too heavily on any single input.

- **Ridge regression** adds a penalty based on the _squared_ size of every coefficient, which shrinks all coefficients down toward zero, but rarely all the way to exactly zero.
- **Lasso regression** adds a penalty based on the _absolute_ size of every coefficient, which can shrink some coefficients all the way down to exactly zero — effectively removing those inputs from the model entirely, acting as a built-in, automatic form of feature selection.

### Seeing It: How Regularization Shrinks Coefficients

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
n, p = 100, 10
X = np.random.normal(0, 1, (n, p))
true_coefs = np.array([5, -3, 2, 0, 0, 0, 1, 0, 0, 0])   # only 4 of the 10 inputs actually matter
y = X @ true_coefs + np.random.normal(0, 1, n)

X_scaled = StandardScaler().fit_transform(X)   # scaling matters here, exactly as taught in Chapter 6, §6.5

alphas = np.logspace(-2, 2, 50)
ridge_coefs = []
lasso_coefs = []

for a in alphas:
    ridge = Ridge(alpha=a).fit(X_scaled, y)
    lasso = Lasso(alpha=a).fit(X_scaled, y)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i in range(p):
    axes[0].plot(alphas, ridge_coefs[:, i])
    axes[1].plot(alphas, lasso_coefs[:, i])

axes[0].set_xscale("log")
axes[0].set_title("Ridge: Coefficients Shrink, but Rarely Hit Exactly Zero")
axes[0].set_xlabel("Regularization Strength (alpha)")
axes[0].set_ylabel("Coefficient Value")

axes[1].set_xscale("log")
axes[1].set_title("Lasso: Coefficients Can Be Pushed All the Way to Zero")
axes[1].set_xlabel("Regularization Strength (alpha)")

plt.tight_layout()
plt.show()
```

**How to read this picture, in simple words:** 
- each colored line follows one single input column's coefficient as the regularization strength (alpha) increases from left to right. On the far left, where alpha is tiny, regularization barely does anything, and every coefficient sits wherever plain, unregularized linear regression would put it. 
- As alpha grows, moving right, every line gets pulled toward zero — the model is being pushed harder and harder to rely less on every input. 
- In the Ridge panel, notice that every single line shrinks _toward_ zero but keeps a life of its own — none of them ever fully flatten out exactly onto the zero line, even at high alpha. In the Lasso panel, watch for lines that visibly flatten completely onto zero and stay there — these correspond exactly to the six columns in this example that were deliberately built to have no real effect on the outcome (their `true_coefs` were exactly 0), and Lasso has correctly, automatically identified and eliminated them, while Ridge merely shrank them without ever fully removing them. 
- **This is the single clearest way to see the real, practical difference between these two techniques**: Ridge shrinks everything a little; Lasso can genuinely throw some inputs away entirely.

### Advantages

- **Extremely fast to train and simple to explain** — a coefficient like "$\beta_1 = 2.5$" translates directly into a plain-English sentence: "for every extra unit of $x$, we predict 2.5 more units of $y$."
- **Directly tells you the size and direction of each input's effect**, which is often just as valuable to a business as the prediction itself.

### Disadvantages

- **Assumes a straight-line relationship** — if the true relationship is curved (like the U-shape from Chapter 5, §5.2.4), plain linear regression will systematically misrepresent it.
- **Sensitive to outliers**, since it's built on squared errors, which — exactly like variance (Chapter 5, §5.1.7) — punish large mistakes disproportionately hard.

### Python Implementation

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

### Business Scenario, Explained Step by Step

A real estate agency wants to help sellers set a realistic asking price for their homes, rather than guessing based on gut feeling. The agency's data team gathers a few years of past sale prices, along with square footage, number of bedrooms, and distance to the nearest school for each home, and trains a linear regression model.

Once trained, the model produces something genuinely useful beyond just predictions: a clear, explainable set of coefficients. Suppose the coefficient for square footage comes out to 120 — this tells the agency, in plain language, "all else being equal, every extra square foot of a house adds about $120 to its predicted sale price." If the coefficient for distance to the nearest school comes out strongly negative, that tells them proximity to schools is a genuinely significant factor buyers are willing to pay for. When a new seller lists their home, the agency plugs in that home's specific numbers and gets back an instant, data-backed suggested price — and, just as importantly, can explain exactly _why_ that price makes sense, pointing directly to the specific factors (size, bedrooms, school distance) driving the number, rather than presenting the prediction as an unexplainable black box.

---

## 8.1.2 Logistic Regression

### Why Not Just Use Linear Regression for a Yes/No Question?

It's tempting to think: "if linear regression predicts a number, and I want to predict something like 'will this customer cancel: yes or no,' I could just treat yes as 1 and no as 0, and use linear regression directly." This turns out to be a genuinely bad idea, and seeing exactly why is one of the most useful things a beginner can learn.

### Seeing It: Why a Straight Line Fails for a Yes/No Question

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(2)
hours_studied = np.concatenate([np.random.uniform(0, 3, 20), np.random.uniform(2, 8, 20)])
passed_exam = np.concatenate([np.zeros(20), np.ones(20)])
# add a bit of realistic overlap
passed_exam[15:20] = np.random.choice([0, 1], 5, p=[0.3, 0.7])

model = LinearRegression().fit(hours_studied.reshape(-1, 1), passed_exam)
x_line = np.linspace(-1, 9, 100)
y_line = model.predict(x_line.reshape(-1, 1))

plt.figure(figsize=(8, 5))
plt.scatter(hours_studied, passed_exam, color="steelblue", label="Actual students (0 = failed, 1 = passed)")
plt.plot(x_line, y_line, color="crimson", linewidth=2, label="Linear regression's line")
plt.axhline(0, color="gray", linewidth=0.5)
plt.axhline(1, color="gray", linewidth=0.5)
plt.ylabel("Passed Exam (0 or 1)")
plt.xlabel("Hours Studied")
plt.title("Why a Straight Line Is the Wrong Tool for a Yes/No Question")
plt.legend()
plt.show()
```

**How to read this picture, in simple words:** every dot is a real student, sitting at exactly 0 (failed) or exactly 1 (passed) on the y-axis — there's no such thing as a student who "0.6 passed." The red line is what plain linear regression produces. Notice two serious problems immediately: first, for a student who studied very few hours, the line dips _below_ zero — predicting a "probability" of passing that's literally negative, which makes no real-world sense at all. Second, for a student who studied a great many hours, the line rises _above_ one — predicting an impossible "110% chance of passing." A straight line, by its very nature, keeps going up (or down) forever in both directions, but a genuine probability must always stay between 0 and 1. **This single picture is the entire reason logistic regression exists**: it's built specifically to keep every prediction trapped between 0 and 1, no matter how extreme the input.

### The Sigmoid Function: The Fix

Logistic regression solves this problem with a clever mathematical trick: instead of predicting the outcome directly with a straight line, it first calculates a straight line as usual, and then squeezes that result through a special S-shaped curve called the **sigmoid function**, which converts any number, no matter how large or small, into a value strictly between 0 and 1.

$$p = \frac{1}{1 + e^{-z}} \qquad \text{where } z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots$$

Here, $z$ is calculated exactly like ordinary linear regression, and $e$ is a specific mathematical constant (approximately 2.718) used throughout science and statistics. The important thing isn't memorizing this formula — it's understanding what it _does_.

### Seeing It: The Sigmoid Curve

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 200)
p = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, p, color="darkorange", linewidth=2)
plt.axhline(0.5, color="gray", linestyle="--", label="0.5 — the usual decision cutoff")
plt.axhline(0, color="lightgray", linewidth=0.8)
plt.axhline(1, color="lightgray", linewidth=0.8)
plt.xlabel("z (the straight-line part of the model)")
plt.ylabel("Predicted Probability")
plt.title("The Sigmoid Function: Squeezing Any Number Into a 0-to-1 Probability")
plt.legend()
plt.show()
```

**How to read this picture, in simple words:** no matter how far left or right you go on the x-axis — even to an extreme value like -1,000 or +1,000 — the curve never actually reaches 0 or 1; it only ever gets closer and closer, hugging those two boundaries without ever crossing them. This is exactly the guarantee logistic regression needed: a genuine, always-valid probability, never negative and never above 100%. The dashed line at 0.5 marks the typical cutoff point: if the sigmoid's output is above 0.5, the model predicts "yes" (or category 1); if it's below 0.5, it predicts "no" (or category 0) — this exact cutoff is the **decision threshold** first introduced in Chapter 5, §5.6.6, and it can be moved up or down depending on whether false alarms or missed cases matter more for the specific business problem at hand.

### Seeing It: The Decision Boundary

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=3)
model = LogisticRegression().fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
plt.colorbar(label="Predicted Probability of Class 1")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="black")
plt.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
plt.title("Logistic Regression's Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

**How to read this picture, in simple words:** the background coloring shows the predicted probability everywhere on the chart, not just at the actual data points — deep red areas mean the model is very confident in class 1, deep blue areas mean it's very confident in class 0, and the pale, washed-out region in between marks genuine uncertainty. The solid black line is drawn exactly where the predicted probability equals 0.5 — this is the actual **decision boundary**, first introduced conceptually in Chapter 5, §5.5.3. Notice that this boundary is a straight line (or, with more input columns, a flat plane) — this is an important, sometimes limiting property of logistic regression: it can only ever separate two classes with a straight cut, never a curved one, unless you deliberately engineer curved or interaction features first (Chapter 6, §6.6.5).

### Hyperparameters

|Hyperparameter|What It Controls|Practical Guidance|
|---|---|---|
|**`C` (inverse regularization strength)**|How much the model is penalized for large coefficients|Smaller `C` means stronger regularization (a simpler model); larger `C` allows the model more freedom to fit the training data closely|
|**`penalty`**|Whether to use Ridge-style (`l2`) or Lasso-style (`l1`) regularization|`l2` is the common default; `l1` if automatic feature elimination is desired, exactly as shown in §8.1.1|
|**Decision threshold**|The cutoff probability used to convert a prediction into a final "yes/no" label|Not always a formal hyperparameter of the model itself, but a genuine, adjustable business decision, exactly as covered in Chapter 5, §5.6.6|

### Advantages

- **Outputs a genuine, meaningful probability**, not just a raw label — extremely useful for business decisions like "only flag this customer if we're at least 80% confident," not just a blunt yes/no.
- **Coefficients remain directly interpretable**, in terms of their effect on the log-odds of the outcome, keeping much of linear regression's explainability.

### Disadvantages

- **Can only draw a straight decision boundary** between classes, unless additional engineered features are added — a genuinely curved, complex relationship between classes can defeat plain logistic regression entirely.
- **Assumes the different input columns don't have extreme, unaddressed multicollinearity** (Chapter 5, §5.2.8), which can make its coefficients unstable and hard to trust.

### Python Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

predicted_labels = model.predict(X_test)
predicted_probabilities = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, predicted_labels))
print("AUC:", roc_auc_score(y_test, predicted_probabilities))
```

Notice that this final evaluation step reuses exactly the tools introduced back in Chapter 5, §5.6 — accuracy, and AUC built from the model's predicted probabilities — directly connecting this chapter's algorithms to that chapter's evaluation toolkit.

### Business Scenario, Explained Step by Step

A telecom company wants to predict which customers are likely to cancel their subscription (churn) within the next month, using data like monthly bill, number of support calls, and contract length. The company chooses logistic regression specifically because the marketing team wants two things at once: a genuine probability for each customer (to prioritize the highest-risk ones first for a retention call) and a model whose coefficients they can actually explain to leadership.

Once trained, the model reveals that the coefficient for "number of support calls in the last 90 days" is strongly positive — meaning more support calls are associated with a meaningfully higher predicted probability of cancellation, exactly the kind of domain-relevant signal discussed as a feature engineering opportunity back in Chapter 6, §6.6.6. The marketing team decides, based on limited budget for retention calls, to only reach out to customers with a predicted cancellation probability above 70% (a deliberately chosen, stricter decision threshold than the default 0.5) — directly applying the precision-recall trade-off first covered in Chapter 5, §5.6.6, prioritizing a smaller number of very high-confidence, high-risk customers rather than casting a wide, less targeted net.

---

## 8.1.3 Linear vs. Logistic Regression: Choosing the Right One

|Situation|Best Choice|Why|
|---|---|---|
|Predicting a number from a wide, continuous range (price, revenue, temperature)|**Linear Regression**|Directly designed to predict a continuous number|
|Predicting a yes/no category, or the probability of one|**Logistic Regression**|Squeezes its output through the sigmoid function to always produce a valid probability between 0 and 1|
|You want directly interpretable coefficients either way|**Either**|Both models produce coefficients that translate into a plain-English sentence about each input's effect|
|The true relationship looks clearly curved, not straight|**Neither, as-is**|Consider engineered features (Chapter 6, §6.6) or a more flexible model (Sections 8.2 onward)|

This exact distinction was first introduced conceptually in Chapter 5, §5.5.1 ("is the thing you're predicting a number or a category?") — this section has now shown precisely _how_ each of those two answers becomes a real, working algorithm.

---

## 8.1.4 Common Mistakes

|Mistake|Why It's a Problem|What To Do Instead|
|---|---|---|
|**Using linear regression to predict a yes/no outcome directly**|Produces nonsensical "probabilities" below 0 or above 1, exactly as shown in §8.1.2's visualization|Use logistic regression for any yes/no or categorical outcome|
|**Setting the learning rate without any testing**|A learning rate that's too small wastes enormous time; one that's too large can fail to converge entirely, exactly as shown in §8.1.1|Try a few different learning rates and watch how quickly and smoothly the cost function decreases|
|**Applying regularization without scaling the input columns first**|Regularization penalizes large coefficients, but a column measured in the thousands will naturally need a smaller coefficient than one measured in single digits, making the penalty unfair across columns|Always scale numeric inputs (Chapter 6, §6.5) before applying Ridge or Lasso regularization|
|**Assuming a positive logistic regression coefficient means strong real-world importance**|A coefficient's size also depends on the scale of its input column, and multicollinearity (Chapter 5, §5.2.8) can distort coefficients in confusing ways|Scale inputs consistently, and check for multicollinearity before over-interpreting individual coefficients|
|**Always using the default 0.5 decision threshold without considering the business cost of false alarms vs. missed cases**|As covered in Chapter 5, §5.6.6, the right threshold depends entirely on which type of mistake is more costly in a specific situation|Deliberately choose a threshold based on real business trade-offs, exactly as the telecom churn example did|

---

## 8.1.5 Best Practices

- **Always visualize your data before choosing between linear and logistic regression** — a quick scatter plot, like the one in §8.1.2, often makes the right choice obvious.
- **Scale your input columns before applying regularization**, so the penalty treats every column fairly (Chapter 6, §6.5).
- **Test a few different learning rates** rather than accepting a default value blindly, watching whether the cost function decreases smoothly or behaves erratically.
- **Use Lasso when you suspect several input columns are genuinely useless**, since it can eliminate them automatically; use Ridge when you believe most columns contribute at least a little.
- **Treat the logistic regression decision threshold as a real, adjustable business decision**, not a fixed technical default, exactly as shown in the telecom churn scenario.

---

## 8.1.6 Interview Notes

**Q1: "Explain, in plain terms, what gradient descent is doing."** A strong answer describes the "walking downhill in the fog" analogy: starting from some guess, repeatedly checking which direction reduces the error, and taking a small step in that direction, until further steps stop meaningfully improving things (§8.1.1).

**Q2: "Why can't you just use linear regression for a binary classification problem?"** A strong answer explains that a straight line has no natural floor or ceiling, so it can predict values below 0 or above 1, which makes no sense for a probability — logistic regression fixes this by passing its output through the sigmoid function, which is mathematically guaranteed to stay between 0 and 1 (§8.1.2).

**Q3: "What is the difference between Ridge and Lasso regression?"** A strong answer explains that both add a penalty for large coefficients to discourage overfitting, but Ridge's penalty (based on squared coefficient size) shrinks coefficients toward zero without usually reaching it, while Lasso's penalty (based on absolute coefficient size) can shrink some coefficients all the way to exactly zero, effectively removing those features (§8.1.1).

**Q4: "What happens if your learning rate is set too high?"** A strong answer explains that each gradient descent step can overshoot past the true minimum of the cost function, causing the model's parameters to bounce around instead of settling smoothly, and in bad cases, to diverge further away from the correct answer entirely (§8.1.1).

**Q5 (Scenario-based): "You're building a model to predict customer churn, and your manager wants to know exactly which factors drive cancellations, in plain language, not just a prediction. Which model would you choose, and why?"** A strong answer proposes logistic regression, explaining that its coefficients translate directly into interpretable statements about each factor's effect on the probability of churn, satisfying the need for both a usable prediction and a genuinely explainable model — directly mirroring the telecom scenario in §8.1.2.

---

## 8.1.7 Section Summary

This section turned the regression and classification concepts from Chapter 5 (§5.5) into real, working algorithms:

- **Linear regression** finds the best-fit straight line by minimizing Mean Squared Error, using **gradient descent** — repeatedly stepping downhill on the cost function until reaching the lowest point.
- The **learning rate** controls how big each gradient descent step is, and getting it wrong (too small or too large) causes real, visible problems.
- **Ridge and Lasso regression** add a penalty for large coefficients to fight overfitting, with Lasso capable of shrinking some coefficients all the way to zero.
- **Logistic regression** solves the "can't use a straight line for yes/no questions" problem by passing a linear model's output through the **sigmoid function**, guaranteeing a valid probability between 0 and 1.
- Logistic regression's **decision boundary** is always a straight line (or flat plane), and its **decision threshold** is a genuine, adjustable business choice, not a fixed default.

> **Where to go next:** Section 8.2 moves to **Decision Trees and Random Forest** — a fundamentally different approach that doesn't rely on straight lines at all, and can naturally capture curved, complex relationships that linear and logistic regression would completely miss.