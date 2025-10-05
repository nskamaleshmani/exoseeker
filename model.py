
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings(action="ignore") 

def run_inference(X_train, y_train, choosen_classifiers,
                  rf_n_estimators, rf_max_depth, 
                  gb_n_estimators, gb_max_depth, 
                  mlp_max_iter, mlp_alpha,
                  random_state=42):
    
    estimators = [("rf", RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, 
                                                    random_state=random_state))]
    
    if "Random Forest" not in choosen_classifiers:
        estimators.append(tuple(("rf", RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, 
                                                    random_state=random_state))))
    if "Gradient Boosting" in choosen_classifiers:
        estimators.append(tuple(("gb", GradientBoostingClassifier(n_estimators=gb_n_estimators, max_depth=gb_max_depth, 
                                                        random_state=random_state))))
    if "Multi-layer Perceptron" in choosen_classifiers:
        estimators.append(tuple(("mlp", MLPClassifier(max_iter=mlp_max_iter, alpha=mlp_alpha, 
                                                                    random_state=random_state))))

    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    model.fit(X_train, y_train)
    
    return model