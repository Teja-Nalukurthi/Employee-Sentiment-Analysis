# Employee Sentiment Analysis Project

## Overview
This project analyzes employee messages to assess sentiment and engagement using Natural Language Processing (NLP) and machine learning techniques. The analysis includes sentiment labeling, exploratory data analysis, employee scoring, ranking, flight risk identification, and predictive modeling.

---

## Project Structure

```
sentiment-analysis/
├── sentiment_analysis.ipynb          # Main analysis notebook
├── test(in).csv                      # Input dataset (unlabeled)
├── labeled_dataset.csv               # Output dataset with sentiment labels
├── top_positive_employees.csv        # Monthly top 3 positive employees
├── top_negative_employees.csv        # Monthly top 3 negative employees
├── flight_risk_employees.csv         # Employees at risk of leaving
├── model_performance.csv             # Predictive model metrics
├── feature_importance.csv            # Model feature coefficients
├── visualizations/                   # Folder containing all charts and graphs
│   ├── 01_sentiment_distribution.png
│   ├── 02_temporal_trends.png
│   ├── 03_employee_activity.png
│   ├── 04_message_characteristics.png
│   ├── 05_word_clouds.png
│   ├── 06_monthly_scores_distribution.png
│   ├── 07_employee_rankings.png
│   ├── 08_flight_risk_analysis.png
│   ├── 09_model_performance.png
│   └── 10_correlation_heatmap.png
├── README.md                         # This file
└── Final_Report.docx                 # Detailed project report
```

---

## Technologies Used

- **Python 3.8+**
- **Libraries:**
  - `pandas`, `numpy` - Data manipulation
  - `matplotlib`, `seaborn` - Data visualization
  - `nltk` (VADER) - Sentiment analysis
  - `transformers`, `textblob` - NLP support
  - `scikit-learn` - Machine learning
  - `wordcloud` - Text visualization
  - `torch` - Deep learning framework

---

## Methodology

### Task 1: Sentiment Labeling
- **Approach:** VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer
- **Classification:**
  - Compound score ≥ 0.05 → **Positive**
  - Compound score ≤ -0.05 → **Negative**
  - Otherwise → **Neutral**
- **Rationale:** VADER is specifically optimized for social media and short text, providing fast and accurate sentiment classification

### Task 2: Exploratory Data Analysis (EDA)
Comprehensive analysis including:
- Dataset structure and completeness
- Sentiment distribution (pie charts, bar charts)
- Temporal trends (monthly, weekly patterns)
- Employee activity patterns
- Message characteristics (length, word count)
- Word clouds for different sentiment categories

### Task 3: Employee Score Calculation
- **Scoring System:**
  - Positive message: +1
  - Negative message: -1
  - Neutral message: 0
- Scores aggregated monthly per employee
- Scores reset at the beginning of each month

### Task 4: Employee Ranking
- **Top 3 Positive Employees:** Highest monthly scores (sorted descending by score, then alphabetically)
- **Top 3 Negative Employees:** Lowest monthly scores (sorted ascending by score, then alphabetically)
- Rankings generated for each month in the dataset

### Task 5: Flight Risk Identification
- **Definition:** Employees with 4+ negative messages in any rolling 30-day period
- **Method:** Sliding window analysis across all employee messages
- **Output:** List of at-risk employees with window details

### Task 6: Predictive Modeling
- **Model:** Linear Regression
- **Features:**
  1. Message frequency (count per month)
  2. Average message length
  3. Average word count
  4. Positive message ratio
  5. Negative message ratio
  6. Average sentiment score
- **Target:** Monthly sentiment score
- **Validation:** 80-20 train-test split with standardized features

---

## Key Findings

### 📊 Sentiment Distribution
(Results will populate after running the notebook)
- **Positive:** X% of messages
- **Negative:** Y% of messages
- **Neutral:** Z% of messages

### 🏆 Top 3 Positive Employees (Latest Month)
(Results will populate after running the notebook)
1. Employee Name - Score: XX
2. Employee Name - Score: XX
3. Employee Name - Score: XX

### ⚠️ Top 3 Negative Employees (Latest Month)
(Results will populate after running the notebook)
1. Employee Name - Score: -XX
2. Employee Name - Score: -XX
3. Employee Name - Score: -XX

### 🚨 Flight Risk Employees
(Results will populate after running the notebook)
- **Total Identified:** XX employees
- **Criteria:** 4+ negative messages in any 30-day period
- **See:** `flight_risk_employees.csv` for complete list

### 📈 Predictive Model Performance
(Results will populate after running the notebook)
- **R² Score:** 0.XXXX (XX% variance explained)
- **RMSE:** X.XX
- **MAE:** X.XX
- **Interpretation:** The model can predict monthly sentiment scores with reasonable accuracy

---

## Recommendations

### Immediate Actions
1. **Flight Risk Mitigation:** Contact identified at-risk employees for one-on-one discussions
2. **Recognition Programs:** Acknowledge and reward consistently positive employees
3. **Support Systems:** Provide additional resources to employees with negative sentiment trends

### Ongoing Monitoring
1. **Monthly Reviews:** Track sentiment trends and update rankings
2. **Early Warning System:** Use the predictive model to forecast potential issues
3. **Intervention Strategies:** Develop targeted programs based on sentiment drivers

### Strategic Initiatives
1. **Engagement Surveys:** Complement sentiment analysis with direct feedback
2. **Team Building:** Focus on departments with lower sentiment scores
3. **Communication Training:** Help employees express concerns constructively
4. **Work Environment:** Address systemic issues identified through sentiment patterns

---

## How to Run

### Prerequisites
```bash
# Python 3.8 or higher
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn transformers torch nltk textblob wordcloud
```

### Execution
1. Ensure `test(in).csv` is in the project directory
2. Open `sentiment_analysis.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially (Cell → Run All)
4. Review generated outputs, visualizations, and CSV files

### Expected Runtime
- **Small dataset (<10K messages):** ~5-10 minutes
- **Medium dataset (10K-100K messages):** ~15-30 minutes
- **Large dataset (>100K messages):** ~30-60 minutes

---

## Output Files

### CSV Files
- `labeled_dataset.csv` - Original dataset with added sentiment labels
- `top_positive_employees.csv` - Monthly rankings of positive employees
- `top_negative_employees.csv` - Monthly rankings of negative employees
- `flight_risk_employees.csv` - Employees identified as flight risks
- `model_performance.csv` - Model evaluation metrics
- `feature_importance.csv` - Feature coefficients from regression model

### Visualizations
All charts are saved in the `visualizations/` folder:
1. Sentiment distribution (pie and bar charts)
2. Temporal trends (time series, day of week)
3. Employee activity patterns
4. Message characteristics by sentiment
5. Word clouds (positive, negative, neutral)
6. Monthly score distributions
7. Employee rankings visualization
8. Flight risk analysis charts
9. Model performance plots
10. Feature correlation heatmap

---

## Model Insights

### Most Influential Features
(Based on linear regression coefficients)
1. **Positive/Negative Ratios:** Strongest predictors of monthly scores
2. **Average Sentiment Score:** Highly correlated with overall performance
3. **Message Frequency:** Moderate influence on sentiment trends
4. **Message Characteristics:** Length and word count provide additional context

### Limitations
- Sentiment analysis accuracy depends on message quality and context
- Sarcasm and nuanced language may be misclassified
- Cultural and linguistic variations may affect results
- Model assumes consistent employee behavior patterns

### Future Enhancements
- Incorporate neural network models (BERT, RoBERTa) for improved accuracy
- Add topic modeling to identify specific concerns
- Include temporal features (trends, seasonality)
- Develop real-time monitoring dashboard
- Cross-reference with HR data (tenure, department, role)

---

## Contact

**Project Author:** Tejan  
**Submission Date:** February 2026  
**Email:** [Your Email]  
**GitHub:** [Your GitHub Repository]

---

## License

This project is developed as part of an AI assignment for Glynac AI.

---

## Acknowledgments

- Dataset source: Employee email communications
- Sentiment Analysis: NLTK VADER lexicon
- Machine Learning: scikit-learn library
- Visualization: matplotlib, seaborn, wordcloud

---

**Note:** This README will be automatically updated with specific results after running the sentiment analysis notebook. For detailed methodology, findings, and interpretations, please refer to `Final_Report.docx`.
