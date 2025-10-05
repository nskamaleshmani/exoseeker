import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_dataset(df, target):
    if target is False:  
        df = df.copy()
    
        df = df.drop(["loc_rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score"], axis=1)
        
        false_positive_rows = df.query("koi_disposition == 'FALSE POSITIVE'").index
        df = df.drop(false_positive_rows, axis=0).reset_index(drop=True)
        
        df["koi_tce_delivname"] = df["koi_tce_delivname"].fillna(df["koi_tce_delivname"].mode()[0])
        for column in df.columns[df.isna().sum() > 0]:
            df[column] = df[column].fillna(df[column].mean())
        
        df = df.drop(["koi_teq_err1", "koi_teq_err2", "koi_tce_delivname"], axis=1)  
        
        y = df["koi_disposition"]
        X = df.drop("koi_disposition", axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            
        return X_train, X_test, y_train, y_test
    elif target is True:
        df = df.copy()
    
        df = df.drop(["loc_rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score"], axis=1)
        
        df["koi_tce_delivname"] = df["koi_tce_delivname"].fillna(df["koi_tce_delivname"].mode()[0])
        for column in df.columns[df.isna().sum() > 0]:
            df[column] = df[column].fillna(df[column].mean())
        
        df = df.drop(["koi_teq_err1", "koi_teq_err2", "koi_tce_delivname"], axis=1)  
        
        X = df
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        
        return X