
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import json

class LegalAidPreprocessor:
    def __init__(self):
        self.encoders = {}
        self.config = {
            'min_support': 0.025,
            'min_confidence': 0.55,
            'min_lift': 1.3,
            'resolution_time_categories': {
                1: '≤3 mo',
                2: '3–6 mo', 
                3: '6–12 mo',
                4: '12–24 mo',
                5: '>24 mo'
            }
        }
    
    def preprocess_data(self, df):
        """Main preprocessing pipeline"""
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Step 2: Encode categorical features
        df_processed = self._encode_features(df_processed)
        
        # Step 3: Create transaction format for ARM
        transactions = self._create_transactions(df_processed)
        
        return df_processed, transactions
    
    def _handle_missing_values(self, df):
        """Encode missing values as explicit categories"""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                # For numeric columns with missing values
                if df[col].isnull().any():
                    df[col] = df[col].fillna(-1)
        return df
    
    def _encode_features(self, df):
        """Encode categorical features"""
        categorical_cols = [
            'Age_Group', 'Gender', 'Income_Bracket', 'Vulnerability_Category',
            'Case_Type', 'Case_Sub_Type', 'Jurisdiction_Level', 'Legal_Aid_Modality',
            'State', 'Resolution_Time'
        ]
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # For unseen categories during inference
                unseen_mask = ~df[col].isin(self.encoders[col].classes_)
                if unseen_mask.any():
                    # Handle unseen categories
                    df.loc[unseen_mask, col] = 'Unknown'
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        # Convert hearing count to categories
        df['Hearing_Count_Cat'] = pd.cut(df['Hearing_Count'], 
                                         bins=[0, 3, 6, 10, 15],
                                         labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # Convert adjournment count to categories
        df['Adjournment_Count_Cat'] = pd.cut(df['Adjournment_Count'],
                                             bins=[-1, 0, 2, 5, 10],
                                             labels=['None', 'Low', 'Medium', 'High'])
        
        return df
    
    def _create_transactions(self, df):
        """Convert DataFrame to transaction format for ARM"""
        transactions = []
        
        for _, row in df.iterrows():
            transaction = []
            
            # Add demographic features
            transaction.append(f"Age_Group={row['Age_Group']}")
            transaction.append(f"Gender={row['Gender']}")
            transaction.append(f"Income_Bracket={row['Income_Bracket']}")
            transaction.append(f"Vulnerability={row['Vulnerability_Category']}")
            
            # Add case characteristics
            transaction.append(f"Case_Type={row['Case_Type']}")
            transaction.append(f"Jurisdiction={row['Jurisdiction_Level']}")
            transaction.append(f"Aid_Modality={row['Legal_Aid_Modality']}")
            
            # Add procedural features
            transaction.append(f"State={row['State']}")
            transaction.append(f"Hearing_Count={row['Hearing_Count_Cat']}")
            transaction.append(f"Adjournment_Count={row['Adjournment_Count_Cat']}")
            transaction.append(f"Resolution_Time={row['Resolution_Time']}")
            
            # Add outcome flags if they exist
            if 'Delay_Flag' in row and row['Delay_Flag'] == 1:
                transaction.append("OUTCOME=DELAY")
            if 'Outcome_Disparity' in row and row['Outcome_Disparity'] == 1:
                transaction.append("OUTCOME=UNFAVORABLE")
            if 'Aid_Disparity' in row and row['Aid_Disparity'] == 1:
                transaction.append("OUTCOME=AID_DISPARITY")
            
            transactions.append(transaction)
        
        return transactions
    
    def save_config(self, path='preprocessor_config.pkl'):
        """Save preprocessing configuration"""
        with open(path, 'wb') as f:
            pickle.dump({'encoders': self.encoders, 'config': self.config}, f)
    
    def load_config(self, path='preprocessor_config.pkl'):
        """Load preprocessing configuration"""
        with open(path, 'rb') as f:
            saved = pickle.load(f)
            self.encoders = saved['encoders']
            self.config = saved['config']