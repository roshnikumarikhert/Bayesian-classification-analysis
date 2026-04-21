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

I undertook this analysis to investigate how a model forms structured belief from data, not as a computational trick, but as a principled reasoning process. I wanted to examine classification at the level where assumptions, uncertainty, and evidence interact, and to understand how a model behaves when the world it encounters is messier than the theory that defines it.
Bayesian methods provided the ideal lens for this. They force every step of the workflow to be explicit:
how prior expectations are set, how likelihoods reshape those expectations, and how each feature contributes to the final posterior decision. Instead of relying on opaque optimisation, this approach allowed me to study the architecture of inference itself.

This work was driven by several deeper questions:
- How does a model reconcile conflicting or imperfect signals?
- What happens when theoretical assumptions collide with real‑world data structure?
- Which features meaningfully shift the model’s internal belief state, and why?
- Where does uncertainty originate, and how does it propagate through the pipeline?
  
By building the entire workflow end‑to‑end, from data preparation and feature diagnostics to posterior interpretation, I could observe how each analytical decision shapes the model’s reasoning. This process helped me move beyond surface‑level performance metrics and focus instead on interpretability, structural clarity, and the causal logic behind predictions.
Ultimately, I did this to strengthen my ability to think in a statistically disciplined way: to interrogate assumptions, trace information flow, and understand not just what a model predicts, but why it arrives at that conclusion.
It reflects an interest in modelling that prioritises transparency, principled analysis, and rigorous reasoning over complexity for its own sake.


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
The results of this work show that predictive strength often emerges not from model complexity, but from the alignment between data structure and statistical assumptions. By tracing how likelihoods shift across features and how posterior beliefs evolve, the workflow reveals that Gaussian Naive Bayes remains effective even when independence conditions are imperfect, provided that the dataset contains a few dominant, high‑signal variables.
The evaluation metrics expose asymmetries that reflect underlying patterns in the data rather than simple model shortcomings. Accuracy, recall, F1 score, and the confusion matrix together illustrate that understanding a classifier requires examining how information flows through the pipeline, how uncertainty is distributed, and where the model’s reasoning aligns with or diverges from the true structure of the dataset.
What emerges is a broader insight: transparent, statistically principled models retain significant value because they make their internal logic visible. When supported by disciplined preprocessing and careful feature analysis, they offer not only reliable predictions but a coherent explanation of why those predictions arise, a quality essential for analytical work that prioritises interpretability, robustness, and sound statistical reasoning.

