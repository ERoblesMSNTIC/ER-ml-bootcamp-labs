import shap
import lime 
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

def analisis_shap_xgboost(model, X_test):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)
    shap_values = explanation.shap_values
    
    #np.abs(shap_values.sum(axis=1)) + explanation.base_values
    
    print(shap_values)

    shap.plots.beeswarm(explanation)

    return(shap_values)
