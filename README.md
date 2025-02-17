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



Code

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from scipy import stats
warnings.filterwarnings('ignore')
  
# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 
  
# metadata 
print(default_of_credit_card_clients.metadata) 
  
# variable information 
print(default_of_credit_card_clients.variables) 

df = pd.concat([X, y], axis=1)

print("1. Dataset Overview")
print("-" * 50)
print("\nShape of the dataset:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

print("\n2. Missing Values Analysis")
print("-" * 50)
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0] if len(missing_values[missing_values > 0]) > 0 else "No missing values found")

print("\n3. Statistical Summary")
print("-" * 50)
print(df.describe())

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation.png')
plt.close()

print("\n4. Feature-specific Analysis")
print("-" * 50)

print("\nDemographic Variables Analysis:")
print("\nGender Distribution (1 = male, 2 = female):")
print(df['X2'].value_counts(normalize=True))

print("\nEducation Distribution (1 = graduate, 2 = university, 3 = high school, 4 = others):")
print(df['X3'].value_counts(normalize=True))

print("\nMarital Status Distribution (1 = married, 2 = single, 3 = others):")
print(df['X4'].value_counts(normalize=True))

print("\nAge Statistics:")
print(df['X5'].describe())

print("\nPayment Status Distribution:")
payment_cols = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11']
for col in payment_cols:
    print(f"\n{col} (Repayment Status):")
    print(df[col].value_counts().sort_index())

bill_amount_cols = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17']
print("\nBill Amount Statistics:")
print(df[bill_amount_cols].describe())

pay_amount_cols = ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']
print("\nPayment Amount Statistics:")
print(df[pay_amount_cols].describe())

print("\n5. Class Balance Analysis")
print("-" * 50)
print("\nDefault payment distribution:")
print(df['Y'].value_counts(normalize=True))


def detect_outliers(df, columns):
    outliers_summary = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outliers_summary[column] = len(outliers)
    return outliers_summary

numerical_cols = ['X1', 'X5'] + ['X12', 'X13', 'X14', 'X15', 'X16', 'X17'] + ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']
outliers = detect_outliers(df, numerical_cols)

print("\n6. Outlier Analysis")
print("-" * 50)
print("\nNumber of outliers per numerical column:")
for col, count in outliers.items():
    print(f"{col}: {count} outliers")

print("\n7. Scale Analysis")
print("-" * 50)
numerical_summary = df[numerical_cols].describe().T
numerical_summary['range'] = numerical_summary['max'] - numerical_summary['min']
print("\nScale analysis for numerical variables:")
print(numerical_summary[['min', 'max', 'range']])

class CreditDefaultAgent:
    def __init__(self, n_bins=3):
        self.n_bins = n_bins
        self.discretizers = {}
        self.model = None
        self.inference = None
        
    def create_features(self, df):
        """Create aggregate features to reduce network complexity"""
        df_new = df.copy()
        
        # Create aggregate features
        # Average payment delay (X6-X11)
        payment_cols = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11']
        df_new['AVG_PAYMENT_DELAY'] = df_new[payment_cols].mean(axis=1)
        
        # Average bill amount (X12-X17)
        bill_cols = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17']
        df_new['AVG_BILL'] = df_new[bill_cols].mean(axis=1)
        
        # Average payment amount (X18-X23)
        pay_cols = ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']
        df_new['AVG_PAYMENT'] = df_new[pay_cols].mean(axis=1)
        
        # Payment ratio
        df_new['PAYMENT_RATIO'] = df_new['AVG_PAYMENT'] / df_new['AVG_BILL'].replace(0, 1)
        
        # Recent payment status
        df_new['RECENT_PAYMENT_STATUS'] = df_new['X6']  # Most recent payment status
        
        # Select relevant features
        selected_features = [
            'X1',           # Credit limit
            'X2', 'X3', 'X4', 'X5',  # Demographics (sex, education, marriage, age)
            'RECENT_PAYMENT_STATUS',  # Most recent payment status
            'AVG_PAYMENT_DELAY',      # Payment delay pattern
            'AVG_BILL',              # Average bill amount
            'AVG_PAYMENT',           # Average payment amount
            'PAYMENT_RATIO'          # Payment to bill ratio
        ]
        
        return df_new[selected_features]
    
    def prepare_data(self, df, fit=True):
        """Prepare data for Bayesian Network"""
        df_discrete = df.copy()
        
        # Discretize continuous variables
        continuous_features = ['X1', 'X5', 'AVG_BILL', 'AVG_PAYMENT', 
                             'PAYMENT_RATIO', 'AVG_PAYMENT_DELAY']
        
        for column in continuous_features:
            if column in df.columns:
                if fit:
                    discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
                    df_discrete[column] = discretizer.fit_transform(df[column].values.reshape(-1, 1)).ravel()
                    self.discretizers[column] = discretizer
                else:
                    if column in self.discretizers:
                        df_discrete[column] = self.discretizers[column].transform(df[column].values.reshape(-1, 1)).ravel()
        
        # Ensure all variables are discrete and of integer type
        for column in df_discrete.columns:
            df_discrete[column] = df_discrete[column].astype(int)
        
        return df_discrete
    
    def define_network_structure(self):
        """Define simplified Bayesian Network structure"""
        edges = [
            # Demographic influences
            ('X2', 'Y'),    # Gender
            ('X3', 'Y'),    # Education
            ('X4', 'Y'),    # Marriage
            ('X5', 'Y'),    # Age
            
            # Financial influences
            ('X1', 'PAYMENT_RATIO'),  # Credit limit
            ('X1', 'Y'),
            
            # Payment behavior
            ('RECENT_PAYMENT_STATUS', 'Y'),
            ('AVG_PAYMENT_DELAY', 'Y'),
            ('PAYMENT_RATIO', 'Y'),
            
            # Bill and payment relationships
            ('AVG_BILL', 'PAYMENT_RATIO'),
            ('AVG_PAYMENT', 'PAYMENT_RATIO'),
            ('AVG_BILL', 'Y'),
            ('AVG_PAYMENT', 'Y')
        ]
        
        return BayesianNetwork(edges)
    
    def fit(self, X, y):
        """Train the Bayesian Network"""
        # Create aggregate features
        X_processed = self.create_features(X)
        
        # Combine with target
        if isinstance(y, np.ndarray):
            y = pd.Series(y.ravel(), name='Y')
        data = pd.concat([X_processed, y], axis=1)
        
        # Prepare data
        print("Preparing data...")
        data_discrete = self.prepare_data(data, fit=True)
        
        # Define network structure
        print("Defining network structure...")
        self.model = self.define_network_structure()
        
        # Fit the model
        print("Estimating CPDs...")
        self.model.fit(data_discrete, estimator=MaximumLikelihoodEstimator)
        
        # Initialize inference engine
        self.inference = VariableElimination(self.model)
        
        return self
    
    def predict(self, X):
        """Predict default probability for new data"""
        # Create aggregate features
        X_processed = self.create_features(X)
        
        # Prepare data
        X_discrete = self.prepare_data(X_processed, fit=False)
        
        predictions = []
        for _, instance in X_discrete.iterrows():
            evidence = {col: int(instance[col]) for col in X_discrete.columns}
            try:
                result = self.inference.query(variables=['Y'], evidence=evidence)
                pred = 1 if result.values[1] > result.values[0] else 0
            except:
                pred = 0
            predictions.append(pred)
        
        return np.array(predictions)

def evaluate_agent():
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the agent
    print("Initializing Credit Default Agent...")
    agent = CreditDefaultAgent(n_bins=3)
    
    print("Training the agent...")
    agent.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = agent.predict(X_test)
    
    # Evaluate performance
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    return agent, accuracy

# Run evaluation
agent, accuracy = evaluate_agent()

def analyze_variable_distributions(data, variable_name):
    """Analyze the distribution of a variable in the data"""
    print(f"\nAnalysis for {variable_name}:")
    print("-" * 40)
    
    # Get value counts and convert to probabilities
    value_counts = data[variable_name].value_counts()
    probabilities = value_counts / len(data)
    
    print("Probability Distribution:")
    for value, prob in probabilities.items():
        print(f"Value {value}: {prob:.4f}")
    
    print(f"\nUnique values: {len(value_counts)}")
    print(f"Most common value: {value_counts.index[0]} (P={probabilities.iloc[0]:.4f})")
    print(f"Least common value: {value_counts.index[-1]} (P={probabilities.iloc[-1]:.4f})")

def calculate_conditional_probabilities(data, target, feature):
    """Calculate conditional probabilities P(target|feature)"""
    print(f"\nConditional Probabilities P({target}|{feature}):")
    print("-" * 40)
    
    # Calculate joint probabilities
    joint_counts = pd.crosstab(data[feature], data[target])
    joint_probs = joint_counts.div(joint_counts.sum().sum())
    
    # Calculate conditional probabilities
    cond_probs = joint_counts.div(joint_counts.sum(axis=1), axis=0)
    
    print("\nConditional Probability Table:")
    print(cond_probs)
    
    # Find strongest relationships
    max_prob = cond_probs.max().max()
    max_indices = np.where(cond_probs == max_prob)
    
    print(f"\nStrongest relationship:")
    print(f"P({target}={max_indices[1][0]}|{feature}={max_indices[0][0]}) = {max_prob:.4f}")

def analyze_network_probabilities(data, agent):
    """Analyze probabilities in the Bayesian Network"""
    print("Analyzing Network Probabilities")
    print("=" * 50)
    
    # Analyze target variable
    print("\nTarget Variable Analysis:")
    print("-" * 40)
    analyze_variable_distributions(data, 'Y')
    
    key_features = [
        'X1',           # Credit limit
        'X2',           # Gender
        'X3',           # Education
        'X4',           # Marriage
        'X5',           # Age
        'AVG_PAYMENT_DELAY',  # Average payment delay
        'AVG_BILL',          # Average bill amount
        'AVG_PAYMENT',       # Average payment amount
        'PAYMENT_RATIO'      # Payment to bill ratio
    ]
    
    print("\nFeature Analysis:")
    for feature in key_features:
        if feature in data.columns:
            print("\n" + "=" * 50)
            print(f"Analysis for {feature}")
            
            # Analyze feature distribution
            analyze_variable_distributions(data, feature)
            
            # Calculate conditional probabilities with target
            calculate_conditional_probabilities(data, 'Y', feature)
            
            # Additional analysis for specific feature types
            if feature in ['AVG_PAYMENT_DELAY', 'PAYMENT_RATIO']:
                print(f"\nDetailed statistics for {feature}:")
                print(data[feature].describe())

    print("\nFeature Relationships:")
    print("=" * 50)
    key_relationships = [
        ('AVG_BILL', 'PAYMENT_RATIO'),
        ('AVG_PAYMENT', 'PAYMENT_RATIO'),
        ('X1', 'PAYMENT_RATIO')  # Credit limit vs payment ratio
    ]
    
    for feature1, feature2 in key_relationships:
        if feature1 in data.columns and feature2 in data.columns:
            correlation = data[feature1].corr(data[feature2])
            print(f"\nCorrelation between {feature1} and {feature2}: {correlation:.4f}")
            
            # Calculate conditional probabilities if both features are discrete
            if data[feature1].nunique() <= 10 and data[feature2].nunique() <= 10:
                calculate_conditional_probabilities(data, feature2, feature1)

    print("\nNetwork Statistics:")
    print("=" * 50)
    
    # Count number of parameters
    total_parameters = 0
    for node in agent.model.nodes():
        if node in data.columns:
            cpd = agent.model.get_cpds(node)
            if cpd is not None:
                total_parameters += np.prod(cpd.values.shape)
    
    print(f"Total number of parameters: {total_parameters}")
    print(f"Number of nodes: {len(agent.model.nodes())}")
    print(f"Number of edges: {len(agent.model.edges())}")
    
    return {
        'target_distribution': data['Y'].value_counts(normalize=True).to_dict(),
        'feature_distributions': {
            feature: data[feature].value_counts(normalize=True).to_dict()
            for feature in key_features if feature in data.columns
        },
        'network_stats': {
            'total_parameters': total_parameters,
            'num_nodes': len(agent.model.nodes()),
            'num_edges': len(agent.model.edges())
        }
    }

if agent.model and hasattr(agent, 'model'):
    print("Preparing data for probability analysis...")
    
    X_processed = agent.create_features(X)
    
    discrete_data = agent.prepare_data(X_processed, fit=False)
    
    discrete_data['Y'] = y
    
    probability_analysis = analyze_network_probabilities(discrete_data, agent)
    
    print("\nSummary of Key Findings:")
    print("=" * 50)
    print("\n1. Target Distribution:")
    for value, prob in probability_analysis['target_distribution'].items():
        print(f"Class {value}: {prob:.4f}")
    
    print("\n2. Most Influential Features:")
    for feature, dist in probability_analysis['feature_distributions'].items():
        if len(dist) <= 5:  # Only show compact distributions
            print(f"\n{feature}:")
            for value, prob in dist.items():
                print(f"  Value {value}: {prob:.4f}")
    
    print("\n3. Network Complexity:")
    stats = probability_analysis['network_stats']
    print(f"Total parameters: {stats['total_parameters']}")
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
else:
    print("Error: Model not properly initialized. Please train the agent first.")
