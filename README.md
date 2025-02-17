# CSE150A Project- Identifying Credit Card Defaults

## Data Exploration 

## 1. Dataset Overview

Shape of the dataset: (30000, 24)

First few rows:
```
       X1  X2  X3  X4  X5  X6  X7  X8  X9  X10  ...    X15    X16    X17  \
0   20000   2   2   1  24   2   2  -1  -1   -2  ...      0      0      0   
1  120000   2   2   2  26  -1   2   0   0    0  ...   3272   3455   3261   
2   90000   2   2   2  34   0   0   0   0    0  ...  14331  14948  15549   
3   50000   2   2   1  37   0   0   0   0    0  ...  28314  28959  29547   
4   50000   1   2   1  57  -1   0  -1   0    0  ...  20940  19146  19131   

    X18    X19    X20   X21   X22   X23  Y  
0     0    689      0     0     0     0  1  
1     0   1000   1000  1000     0  2000  1  
2  1518   1500   1000  1000  1000  5000  0  
3  2000   2019   1200  1100  1069  1000  0  
4  2000  36681  10000  9000   689   679  0  
```

Dataset Info:
- 30,000 entries
- 24 columns
- All columns are of type int64
- Memory usage: 5.5 MB

## 2. Missing Values Analysis

No missing values found in any column.

## 3. Statistical Summary

Key statistics for all features (showing mean, std, min, max, quartiles).

For example, X1 (Credit Amount):
- Mean: 167,484.32
- Std: 129,747.66
- Min: 10,000
- Max: 1,000,000
- Median: 140,000

## 4. Feature-specific Analysis

### Demographic Variables

Gender Distribution (1 = male, 2 = female):
- Female: 60.37%
- Male: 39.63%

Education Distribution:
- University: 46.77%
- Graduate: 35.28%
- High School: 16.39%
- Others: 1.56%

Marital Status:
- Single: 53.21%
- Married: 45.53%
- Others: 1.08%

Age Statistics:
- Mean: 35.49 years
- Median: 34 years
- Range: 21-79 years

### Payment Status Distribution

Repayment Status (X6-X11) shows varying patterns of payment behavior, with most accounts being current (0) or having slight delays (-1, -2).

### Bill Amount Statistics

Bill amounts (X12-X17) show significant variation:
- Maximum bills range from 891,586 to 1,664,089
- Negative values present, indicating possible credits
- Median bills range from 17,071 to 22,381

### Payment Amount Statistics

Payment amounts (X18-X23):
- Most payments concentrated in lower ranges
- Significant outliers present
- Median payments around 1,500-2,100

## 5. Class Balance Analysis

Default payment distribution:
- No Default (0): 77.88%
- Default (1): 22.12%

## 6. Outlier Analysis

Number of outliers detected:
- Credit Amount (X1): 167 outliers
- Age (X5): 272 outliers
- Bill Amounts (X12-X17): ~2,400-2,725 outliers
- Payment Amounts (X18-X23): ~2,700-3,000 outliers

## 7. Scale Analysis

Variable ranges vary significantly:
- Credit Amount: 10,000 to 1,000,000
- Age: 21 to 79
- Bill Amounts: -339,603 to 1,664,089
- Payment Amounts: 0 to 1,684,259

# CPTs

# Network Probability Analysis

## Target Variable Analysis (Y)

Distribution:
- No Default (0): 77.88%
- Default (1): 22.12%

## Feature Analysis

### X1 (Credit Amount)
Distribution:
- Level 2: 37.17%
- Level 1: 32.06%
- Level 0: 30.77%

Strongest relationship: P(No Default|X1=2) = 84.87%

### X2 (Gender)
Distribution:
- Female (2): 60.37%
- Male (1): 39.63%

Strongest relationship: P(No Default|X2=1) = 79.22%

### X3 (Education)
Distribution:
- University (2): 46.77%
- Graduate (1): 35.28%
- High School (3): 16.39%
- Others: 1.51%

Strongest relationship: P(No Default|X3=0) = 100%

### X4 (Marital Status)
Distribution:
- Single (2): 53.21%
- Married (1): 45.53%
- Others (3): 1.08%
- Unknown (0): 0.18%

Strongest relationship: P(No Default|X4=0) = 90.74%

### X5 (Age Group)
Distribution:
- Group 1: 34.28%
- Group 2: 33.66%
- Group 0: 32.06%

Strongest relationship: P(No Default|X5=1) = 79.80%

### Average Payment Delay
Distribution:
- High (2): 57.75%
- Low (0): 30.64%
- Medium (1): 11.61%

Statistics:
- Mean: 1.27
- Median: 2.00
- Std: 0.90

Strongest relationship: P(No Default|AVG_PAYMENT_DELAY=1) = 84.47%

### Average Bill Amount
Distribution:
- High (2): 33.49%
- Low (0): 33.29%
- Medium (1): 33.22%

Strongest relationship: P(No Default|AVG_BILL=2) = 79.19%

### Average Payment
Distribution:
- High (2): 33.49%
- Medium (1): 33.30%
- Low (0): 33.20%

Strongest relationship: P(No Default|AVG_PAYMENT=2) = 86.09%

### Payment Ratio
Distribution:
- High (2): 33.40%
- Low (0): 33.33%
- Medium (1): 33.27%

Statistics:
- Mean: 1.00
- Median: 1.00
- Std: 0.82

Strongest relationship: P(No Default|PAYMENT_RATIO=2) = 84.69%

## Feature Relationships

### Average Bill vs Payment Ratio
- Correlation: -0.5518
- Strongest relationship: P(PAYMENT_RATIO=2|AVG_BILL=0) = 69.99%

### Average Payment vs Payment Ratio
- Correlation: 0.1187
- Strongest relationship: P(PAYMENT_RATIO=2|AVG_PAYMENT=2) = 44.04%

### Credit Amount vs Payment Ratio
- Correlation: 0.1864
- Strongest relationship: P(PAYMENT_RATIO=1|X1=0) = 52.85%

## Network Statistics
- Total parameters: 898,248
- Nodes: 11
- Edges: 13


# Question and Answer

- Explain what your AI agent does in terms of PEAS. What is the "world" like? 

Ans. Using the PEAS model, the performance metric is the accuracy for whether a customer defaulted on their credit card. It is based in a fully observable, deterministic, episodic, semi-dynamic, continuous and single agent environment. The actuator for this agent is the screen display which would include questions about their financial data and the prediction. The sensor for this agent is the keyboard which the user will use to enter the information, run the agent and give it the necessary information. Using all of this information, I plan to build my goal-based agent. The world invloves using past customer demographics, payment history over 6 months, bill amounts over 6 months, payment amounts over 6 months and credit limits. 

- What kind of agent is it? Goal based? Utility based? etc. 

Ans. It is a goal-based agent whose goal it is trying to maximise is to predict with the highest accuracy whether a customer defaulted on their credit card.

- Describe how your agent is set up and where it fits in probabilistic modeling

Ans. I built my agent using a bayesian network which made variables as nodes such as demographics, payment history and bill amounts which represents dependencies through directed edges
and conditional probabilities through CPTs and allows for probabilistic inference for predictions. Each engineered feature has its own conditional probability distribution. It created aggregate features that captured Payment delay patterns, Bill-to-payment ratios and credit utilization patterns. I also discretized the continuous variables as it enables estimation of conditional probabilities. The key probability was P(Default | Evidence) = P(Default | Demographics, PaymentHistory, BillAmounts, PaymentAmounts) wherein I had prior probabilities: Initial default rates, base distributions of demographic variables. Conditional Probabilities:
P(Default | Payment History), P(Default | Bill Amounts), P(Default | Demographics). 

The network structure was 

Demographics (X2-X5) → Default (Y)
                    ↗
Payment History (X6-X11) 
                    ↗
Bill Amounts (X12-X17)
                    ↗
Payment Amounts (X18-X23)


- Train your first model and Evaluate your model

Ans. Here is the output for it:

Training the agent...
Preparing data...
Defining network structure...
Estimating CPDs...

Making predictions...

Model Performance:

Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.95      0.88      4687
           1       0.56      0.25      0.34      1313

    accuracy                           0.79      6000
   macro avg       0.69      0.60      0.61      6000
weighted avg       0.76      0.79      0.76      6000


Confusion Matrix:
[[4435  252]
 [ 988  325]]

Accuracy: 0.7933

- Conclusion 

Ans. In conclusion, my first goal based agent was created based on a Bayesian network that aims to maximise the accuracy of identifying when a user will default on their credit card using the probaility based relationships between the features in the dataset. Using the bayesian network, it acheived an accuraxy of 0.7933. We can improve the model by implementing feature engineering with more sophisticated aggregate features. Another way to improve the model would be to implement adaptive binning based on data distribution or using domain-specific thresholds for certain variables. We could create and use a hybrid approach combining Bayesian Network with other models. We could choose more efficient inference algorithms to find the best accuracy.