# analyzer.py
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

class JusticeGapAnalyzer:
    def __init__(self, min_support=0.025, min_confidence=0.55, min_lift=1.3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = None
        self.frequent_itemsets = None
        
    def fit(self, transactions):
        """Run FP-Growth algorithm"""
        # Convert transactions to one-hot encoded DataFrame
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Mine frequent itemsets using FP-Growth
        self.frequent_itemsets = fpgrowth(df_encoded, 
                                         min_support=self.min_support,
                                         use_colnames=True,
                                         max_len=4)
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            self.rules = association_rules(self.frequent_itemsets, 
                                          metric="confidence",
                                          min_threshold=self.min_confidence)
            
            # Apply lift filtering
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
            
            # Filter for justice gap indicators only
            justice_indicators = ['OUTCOME=DELAY', 'OUTCOME=UNFAVORABLE', 'OUTCOME=AID_DISPARITY']
            self.rules = self.rules[self.rules['consequents'].apply(
                lambda x: any(indicator in str(x) for indicator in justice_indicators)
            )]
            
            # Sort by lift (descending)
            self.rules = self.rules.sort_values('lift', ascending=False)
            
            # Calculate additional metrics
            self._calculate_additional_metrics()
        
        return self.rules
    
    def _calculate_additional_metrics(self):
        """Calculate additional interestingness metrics"""
        if self.rules is not None and len(self.rules) > 0:
            # Calculate conviction
            self.rules['conviction'] = (1 - self.rules['consequent support']) / (1 - self.rules['confidence'])
            
            # Calculate leverage
            self.rules['leverage'] = self.rules['support'] - (
                self.rules['antecedent support'] * self.rules['consequent support']
            )
            
            # Calculate jaccard
            self.rules['jaccard'] = self.rules['support'] / (
                self.rules['antecedent support'] + self.rules['consequent support'] - self.rules['support']
            )
    
    def analyze_justice_gaps(self, df_original, rules_df):
        """Analyze justice gaps based on mined rules"""
        justice_gaps = []
        
        for idx, rule in rules_df.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Extract rule components
            gap_analysis = {
                'rule_id': idx,
                'antecedents': str(antecedents),
                'consequents': str(consequents),
                'support': rule['support'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'gap_type': self._classify_gap_type(antecedents, consequents),
                'policy_implications': self._get_policy_implications(antecedents, consequents)
            }
            
            # Calculate baseline comparison
            if any('OUTCOME=DELAY' in str(c) for c in consequents):
                baseline_delay = df_original['Delay_Flag'].mean()
                gap_analysis['baseline_comparison'] = f"Rule delay rate: {rule['confidence']:.2%} vs Baseline: {baseline_delay:.2%}"
            
            justice_gaps.append(gap_analysis)
        
        return pd.DataFrame(justice_gaps)
    
    def _classify_gap_type(self, antecedents, consequents):
        """Classify the type of justice gap"""
        antecedents_str = str(antecedents).lower()
        consequents_str = str(consequents).lower()
        
        # Demographic disparity
        dem_keywords = ['gender', 'income', 'vulnerability', 'age']
        if any(keyword in antecedents_str for keyword in dem_keywords):
            return "Demographic Disparity"
        
        # Regional disparity
        if 'state' in antecedents_str:
            return "Regional Disparity"
        
        # Case-type bias
        if 'case_type' in antecedents_str:
            return "Case-Type Bias"
        
        # Process bottleneck
        proc_keywords = ['hearing', 'adjournment', 'resolution']
        if any(keyword in antecedents_str for keyword in proc_keywords):
            return "Process-Stage Bottleneck"
        
        return "Systemic Inequity"
    
    def _get_policy_implications(self, antecedents, consequents):
        """Generate policy implications based on rule"""
        implications = []
        
        # Convert to strings for easier processing
        ant_str = ' '.join([str(a) for a in antecedents])
        con_str = ' '.join([str(c) for c in consequents])
        
        if 'gender=female' in ant_str.lower() and 'domestic_violence' in ant_str.lower():
            implications.append("Review gender-sensitive procedures in DV cases")
            implications.append("Allocate specialized resources for female DV applicants")
        
        if 'below_poverty' in ant_str.lower() or 'low_income' in ant_str.lower():
            implications.append("Enhance financial assistance mechanisms")
            implications.append("Review means-testing procedures")
        
        if 'rural' in ant_str.lower() or 'state_' in ant_str.lower():
            implications.append("Improve rural legal aid infrastructure")
            implications.append("Increase mobile legal clinics in underserved areas")
        
        if 'delay' in con_str.lower():
            implications.append("Implement case timeline monitoring")
            implications.append("Establish expedited procedures for high-risk cases")
        
        return '; '.join(implications) if implications else "Requires further investigation"
