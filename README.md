# Predicting Goal Scoring in the English Premier League
### An analysis by Ritviik Ravi, Andrew, Aditya, and Yafet
Published May 2026

---

# Table of Contents

1. Introduction  
2. Data Curation and Preprocessing  
   - Dataset Source  
   - Data Loading and Cleaning  
   - Feature Engineering  
3. Exploratory Data Analysis  
   - Home vs Away Goal Scoring  
   - Shots on Target vs Goals Scored  
   - Match Outcomes Across Eras  

---

# 1. Introduction

Football analytics has become increasingly important in modern sports. Clubs, coaches, analysts, broadcasters, fantasy sports players, and betting organizations all rely on statistical models to better understand team performance and match outcomes. In recent years, machine learning has played a growing role in helping analysts identify patterns that may not be immediately visible through traditional statistics alone.

The objective of our project is to analyze historical Premier League data to determine whether match statistics can be used to predict the number of goals scored during a match. Our analysis spans over two decades of EPL matches, beginning with the 2000/01 season and continuing through the 2024/25 season.

The dataset contains detailed information about each match, including:

- goals scored,
- shots,
- shots on target,
- fouls,
- corners,
- yellow cards,
- teams,
- match dates,
- and other in-game statistics.

Our primary research question is:

> Can historical match statistics be used to reliably predict the number of goals scored in an EPL match?

Answering this question has several practical applications. For clubs and coaching staffs, predictive models can help evaluate offensive and defensive efficiency and improve match preparation. For broadcasters and analysts, predictive insights can improve pre-match coverage and statistical storytelling. For fantasy sports and sports betting communities, more accurate predictions can influence decision-making strategies and match expectations.

More broadly, this project allows us to explore how statistical relationships in sports data translate into predictive machine learning models. By combining exploratory data analysis with regression modeling, we aim to identify which match statistics are most strongly associated with goal scoring and evaluate how effectively machine learning models can generalize those relationships to unseen matches.

---

# 2. Data Curation and Preprocessing

## 2.A Dataset Source

The first step in our analysis was identifying a dataset that contained both match outcomes and detailed in-game statistics suitable for machine learning analysis.

We selected the **English Premier League Match Data 2000–2025** dataset from Kaggle, published by marcohuiii and sourced from football-data.co.uk.

The dataset contains:

- every EPL match from the 2000/01 season through the 2024/25 season,
- over 9,000 total matches,
- full-time and half-time scorelines,
- shots,
- shots on target,
- corners,
- fouls,
- yellow cards,
- red cards,
- referee information,
- and team information.

Each row in the dataset represents a single match.

This dataset is particularly well suited for our analysis because it includes both:
1. the final outcomes of matches, and
2. the in-game statistics that may help explain those outcomes.

We loaded the dataset into a pandas DataFrame, which served as the primary structure for all preprocessing, visualization, and machine learning analysis.

---

## 2.B Data Loading and Cleaning

We first imported the required Python libraries and loaded the CSV file into a pandas DataFrame.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import ttest_ind, pearsonr, chi2_contingency

df = pd.read_csv('epl_final.csv')
```

The MatchDate column was originally stored as a string, so we converted it into a datetime object to allow for easier time-based analysis.

```python
df['MatchDate'] = pd.to_datetime(df['MatchDate'])
df['Year'] = df['MatchDate'].dt.year
```

We then examined the size of the dataset.

```python
print(df.shape)
```

Output:

```python
(9380, 23)
```

The dataset contains:

- 9,380 matches,
- 23 columns,
- 25 seasons,
- and 46 unique teams.

This large sample size provides a strong foundation for both statistical analysis and machine learning modeling.

---



## Preview of Dataset

Before beginning analysis, we examined the first few rows of the dataset to better understand its structure and available features.

```python

df.head()

```

### Sample Dataset Output

| MatchDate | Season | HomeTeam | AwayTeam | FullTimeHomeGoals | FullTimeAwayGoals | FullTimeResult | HomeShots | AwayShots |

|---|---|---|---|---|---|---|---|---|

| 2000-08-19 | 2000/01 | Charlton | Man City | 4 | 0 | H | 17 | 8 |

| 2000-08-19 | 2000/01 | Chelsea | West Ham | 4 | 2 | H | 18 | 10 |

| 2000-08-19 | 2000/01 | Coventry | Middlesbrough | 1 | 3 | A | 11 | 14 |

| 2000-08-19 | 2000/01 | Derby | Southampton | 2 | 2 | D | 9 | 11 |

| 2000-08-19 | 2000/01 | Leeds | Everton | 2 | 0 | H | 15 | 7 |

The dataset includes a wide range of offensive and defensive match statistics, which makes it well suited for both exploratory analysis and predictive modeling.

---

## 2.C Feature Engineering

To support our analysis, we created two additional variables:

### TotalGoals
The total number of goals scored in a match.

### GoalDiff
The difference between home and away goals.

```python
df['TotalGoals'] = df['FullTimeHomeGoals'] + df['FullTimeAwayGoals']

df['GoalDiff'] = (
    df['FullTimeHomeGoals']
    - df['FullTimeAwayGoals']
)
```

These engineered features helped us analyze:
- scoring trends,
- offensive efficiency,
- and match outcomes.

We also examined overall dataset statistics.

```python
print('Seasons:', df['Season'].nunique())
print('Teams:', df['HomeTeam'].nunique())
print('Matches:', len(df))
```

Output:

```python
Seasons: 25
Teams: 46
Matches: 9380
```

---

# 3. Exploratory Data Analysis

Before constructing machine learning models, we conducted exploratory data analysis (EDA) to better understand the structure of the dataset and identify relationships between variables.

We focused on three major questions:

1. Do home teams score more goals than away teams?
2. Is there a relationship between shots on target and goals scored?
3. Have match outcome distributions changed across different eras of EPL history?

Each analysis included:
- a statistical hypothesis test,
- a visualization,
- and an interpretation of the results.

---

# 3.A Home vs Away Goal Scoring

One of the most commonly discussed concepts in sports analytics is home-field advantage. We wanted to determine whether home teams score significantly more goals than away teams in the EPL.

## Hypotheses

### Null Hypothesis (H₀)
The mean number of goals scored by home teams equals the mean number of goals scored by away teams.

### Alternative Hypothesis (Hₐ)
Home teams score more goals on average.

Because our dataset contains over 9,000 matches, we used a one-sided Z-test to compare the two means.

---

## Visualization

VISUALIZATION FOR HOME VS AWAY GOALS DISTRIBUTION

---

## Statistical Analysis

```python
# Visualization: distribution of goals
plt.figure(figsize=(8, 5), facecolor='white')
plt.gca().set_facecolor('white')

data = [df['FullTimeHomeGoals'], df['FullTimeAwayGoals']]
colors = ['steelblue', 'crimson']
labels = ['Home Goals', 'Away Goals']

plt.hist(data, bins=range(0, 10), color=colors, label=labels, align='left')

plt.title('Home vs Away Goals Distribution')
plt.xlabel('Goals per Match')
plt.ylabel('Frequency')

plt.xticks(range(0, 9))
plt.legend()

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
```

```python
# Calculate means
mean_home = df['FullTimeHomeGoals'].mean()
mean_away = df['FullTimeAwayGoals'].mean()

# Calculate standard deviations
std_home = df['FullTimeHomeGoals'].std(ddof=1)
std_away = df['FullTimeAwayGoals'].std(ddof=1)

# Sample sizes
n_home = len(df['FullTimeHomeGoals'])
n_away = len(df['FullTimeAwayGoals'])

# Compute Z statistic
z = (
    (mean_home - mean_away)
    /
    np.sqrt((std_home**2 / n_home) + (std_away**2 / n_away))
)

p = 1 - norm.cdf(z)

print("Z-test for comparing means")
print(f"Home goals mean : {mean_home}")
print(f"Away goals mean : {mean_away}")
print(f"z = {z}")
print(f"p = {p}")
```

Output:

```python
Home goals mean : 1.535
Away goals mean : 1.183
z = 19.58
p = 0.0
```

## Results

Our results showed that home teams score an average of 1.535 goals per game, while away teams score an average of 1.183 goals, a difference of roughly one-third of a goal per match.

With a z-value of 19.58 and a p-value effectively equal to zero, we reject the null hypothesis at all conventional significance levels.

The histogram further reinforces this conclusion. Home teams score multiple goals more frequently and fail to score less often than away teams. This confirms the existence of a statistically significant home advantage effect in the EPL over the past 25 seasons.

This finding is important for predictive modeling because it demonstrates that home and away context plays a meaningful role in offensive production.

---

# 3.B Shots on Target vs Goals Scored

Intuitively, teams that place more shots on target should score more goals. We wanted to determine how strong this relationship is statistically.

To do this, we measured the Pearson correlation between shots on target and goals scored.

## Hypotheses

### Null Hypothesis (H₀)
There is no linear correlation between shots on target and goals scored.

### Alternative Hypothesis (Hₐ)
There is a positive linear correlation between shots on target and goals scored.

---

## Visualization

VISUALIZATION FOR SHOTS ON TARGET VS GOALS SCORED

---

## Statistical Analysis

```python
sot = pd.concat([
    df[['HomeShotsOnTarget', 'FullTimeHomeGoals']]
      .rename(columns={
          'HomeShotsOnTarget': 'SoT',
          'FullTimeHomeGoals': 'Goals'
      }),

    df[['AwayShotsOnTarget', 'FullTimeAwayGoals']]
      .rename(columns={
          'AwayShotsOnTarget': 'SoT',
          'FullTimeAwayGoals': 'Goals'
      })
])

plt.figure(figsize=(6, 5), facecolor='white')
plt.gca().set_facecolor('white')

sample = sot.sample(2000, random_state=42)

m, b = np.polyfit(sot['SoT'], sot['Goals'], 1)

x = np.linspace(sot['SoT'].min(), sot['SoT'].max(), 100)

plt.scatter(
    sample['SoT'],
    sample['Goals'],
    alpha=0.2,
    s=15,
    color='slateblue'
)

plt.plot(x, m*x+b, color='black', linewidth=2)

plt.title('Shots on Target vs Goals Scored')
plt.xlabel('Shots on Target')
plt.ylabel('Goals Scored')

plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()
```

```python
r, p2 = pearsonr(sot['SoT'], sot['Goals'])

print("Pearson r =", r)
print("r^2 =", r**2)
print("p =", p2)
```

Output:

```python
Pearson r = 0.461
r^2 = 0.213
p = 0.0
```

## Results

The scatterplot reveals a clear positive trend between shots on target and goals scored.

An r-value of 0.461 indicates a moderate positive correlation, while the r² value of 0.213 suggests that shots on target explain approximately 21% of the variance in goals scored.

Although goals remain inherently noisy and unpredictable, this represents a strong single-feature relationship for sports data.

The remaining unexplained variance likely comes from:
- shot quality,
- defensive pressure,
- goalkeeper performance,
- tactical adjustments,
- and randomness.

These findings strongly support the inclusion of:
- HomeShotsOnTarget
- and AwayShotsOnTarget

as primary predictive features in future regression models.

---

# 3.C Match Outcomes Across Eras

Football has evolved significantly over the past 25 years. Tactical innovations, changes in officiating, analytics adoption, and even external events such as COVID-19 may have influenced the distribution of match outcomes.

We investigated whether the proportions of:
- home wins,
- draws,
- and away wins

changed significantly across different eras of EPL history.

---

## Hypotheses

### Null Hypothesis (H₀)
Match outcomes are independent of era.

### Alternative Hypothesis (Hₐ)
At least one era has a significantly different outcome distribution.

We divided the dataset into five historical eras and used a chi-squared test on the contingency table.

---

## Visualization

VISUALIZATION FOR MATCH OUTCOME DISTRIBUTIONS ACROSS ERAS

---

## Statistical Analysis

```python
df['Era'] = pd.cut(
    df['Year'],
    bins=[1999, 2005, 2010, 2015, 2019, 2025],
    labels=['2000-05', '2006-10', '2011-15', '2016-19', '2020-25']
)

contingency = pd.crosstab(
    df['Era'],
    df['FullTimeResult']
)[['H', 'D', 'A']]
```

```python
props = contingency.div(
    contingency.sum(axis=1),
    axis=0
) * 100

props.plot(
    kind='bar',
    stacked=True,
    color=['lightskyblue', 'black', 'firebrick']
)

plt.title('Distribution of Match Outcomes Across Eras')
plt.xlabel('Era')
plt.ylabel('Share (%)')

plt.legend(['Home Win', 'Draw', 'Away Win'])

plt.show()
```

```python
chi2, p3, dof, _ = chi2_contingency(contingency)

print("Chi² =", chi2)
print("df =", dof)
print("p =", p3)
```

Output:

```python
Chi² = 28.86
df = 8
p = 0.000335
```

## Results

Since the p-value is well below 0.05, we reject the null hypothesis.

The visualization shows that:
- home win rates remain relatively stable,
- draw rates decline in recent years,
- away wins increase noticeably during the 2020–2025 era.

This trend aligns with the “ghost games” effect during the COVID-19 pandemic, when matches were played without fans and home-field advantage was temporarily reduced.

This suggests that season and era may contain meaningful contextual information for predictive models.
