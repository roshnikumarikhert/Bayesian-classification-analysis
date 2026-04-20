# Bayesian-classification-analysis
## Probabilistic Classification Pipeline using Gaussian Naive Bayes 
### Overview
This analysis develops an end-to-end probabilistic classification pipeline for structured tabular data using a Gaussian Naive Bayes model. The focus extends beyonds implementation to evaluating model assumptions, analysing feature relationships and interpreting predictive behavior within a statistical framework. 

### Motivation 
Rather than defaulting to more compplex models, this workflow investigates whether a statistically grounded, lightweight probabilistic approach can achieve meaningful predictive performance. 

Naive Bayes was deliberately selected to:

- Examine the impact of feature independence assumptions
- Evaluate robustness on real-world structured data
- Provide interpretable probabilistic outputs

### Methodology
#### Data Preparation
- Handled missing values to ensure dataset consistency
- Encoded categorical variables into numerical form
- Prepared structured inputs for statistical modelling

#### Feature Analysis
A key component of the analysis involved assessing feature relationships in the context of Naive Bayes assumptions:

- Computed correlation coefficients across variables
- Analysed inter-feature dependencies
- Identified high-impact features such as sex, pclass, and fare
- This step provides insight into model suitability, not just performance.

#### Exploratory Analysis
- Used boxplots to examine feature distributions
- Detected outliers and distributional differences
- Analysed subgroup behaviour across key variables

#### Model Development
A Gaussian Naive Bayes classifier was implemented due to:

- Computational efficiency
- Suitability for structured tabular data
- Probabilistic interpretability
The dataset was split into training and testing subsets to evaluate generalisation on unseen data.

### Mathematical Framework
$$
P(y \mid x_1, x_2, ..., x_n) = \frac{P(y)\prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, x_2, ..., x_n)}
$$

The model applies Bayes’ theorem to compute posterior probabilities.

$$
P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)
$$

Continuous features are modelled using Gaussian distributions.

$$
\hat{y} = \arg\max_y P(y)\prod_{i=1}^{n} P(x_i \mid y)
$$

The predicted class is selected by maximising posterior probability.

### Model Architecture
![End-to-end probabilistic modelling workflow](model_flow.png)
The workflow illustrates the full pipeline from preprocessing and feature analysis to probabilistic classification and prediction generation.

### Evaluation Strategy
Model performance was assessed using multiple complementary metrics:

- Accuracy – overall classification correctness
- Recall (~0.69) – ability to correctly identify positive instances
- F1 Score (~0.69) – balance between precision and recall
- Confusion Matrix – detailed classification breakdown
- Mean Absolute Error (MAE) – magnitude of prediction error

### Results & Interpretation
The model demonstrated strong predictive capability, but deeper analysis reveals more nuanced behaviour:

- High training accuracy indicates effective pattern capture
- Recall (~0.69) shows the model identifies a substantial proportion of true positives, though some are missed
- The F1 score reflects a balanced trade-off rather than over-optimisation
- The confusion matrix highlights classification asymmetries

#### Key Insight
Naive Bayes remains effective even when its independence assumption is not fully satisfied, provided that key features retain strong predictive signal.

### Limitations
- Feature independence assumption is not strictly met
- Model cannot capture complex feature interactions
- Performance may be constrained compared to more flexible models

### Future Improvements
- Compare performance with Logistic Regression and Random Forest
- Apply feature selection to reduce redundancy
- Introduce cross-validation for more robust evaluation
- Perform hyperparameter tuning

### Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

### Reproducibility
To replicate this analysis:

```bash
pip install -r requirements.txt
jupyter notebook Applied_AI.ipynb
```
### Conclusion
This analysis demonstrates that relatively simple probabilistic models can achieve strong performance when supported by thoughtful preprocessing and analytical evaluation. More importantly, it highlights the importance of understanding model assumptions and interpreting results beyond surface-level accuracy.
