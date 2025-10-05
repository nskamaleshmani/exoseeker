import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.style.use('dark_background')

def model_evaluation(y_test, y_pred):
    accuracy = get_accuracy(*get_classifications(y_test, y_pred)) * 100
    sensitivity = get_sensitivity(*get_classifications(y_test, y_pred)) * 100
    specificity = get_specificity(*get_classifications(y_test, y_pred)) * 100
    percision = get_precision(*get_classifications(y_test, y_pred)) * 100
    f1_score = get_f1_score(*get_classifications(y_test, y_pred)) * 100
    
    metrics_col = ["Accuracy", "Sensitivity", "Specificity", "Percision", "F1 Score"]
    value_col = [accuracy, sensitivity, specificity, percision, f1_score]
    
    eval_results = {"Metrics": metrics_col,
                    "Value":value_col,}
        
    plt.figure(figsize=(10, 8), dpi=400)
    
    fig, ax = plt.subplots()
    
    bar_colors = ["#03045e", "#023e8a", "#0077b6", "#0096c7", "#00b4d8"]

    bars = ax.barh(metrics_col, value_col, label=metrics_col, color=bar_colors, align='center')
    
    formatted_value_col = [f"{accuracy:.2f}%", f"{sensitivity:.2f}%", f"{specificity:.2f}%", f"{percision:.2f}%", f"{(f1_score/100):.4f}"]
        
    ax.bar_label(bars, labels=formatted_value_col, padding=10)
        
    ax.grid(visible=False)
    
    ax.set_xlim(right=110) 
    ax.set_title('Model Metrics')
    ax.legend(title='Fruit color')

    st.pyplot(plt)   

def visualize_confusion_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ["True Neg","False Pos","False Neg","True Pos"]

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(5, 4), dpi=400)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues")

    ax.set_title(f"Confusion Matrix\n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(["False","True"])
    ax.yaxis.set_ticklabels(["False","True"])

    st.pyplot(plt)
    
def get_classifications(y_test, y_pred, positive_label="CONFIRMED"):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    for y_t, y_p in zip(y_test, y_pred):
        if y_t == positive_label:
            if y_p == positive_label:
                tp += 1
            else:
                fn += 1
        else:
            if y_p == positive_label:
                fp += 1
            else:
                tn += 1
    
    return tp, fn, fp, tn

def get_accuracy(tp, fn, fp, tn):
    acc = (tp + tn) / (tp + fn + fp + tn)
    return acc

def get_precision(tp, fn, fp, tn):
    try:
        precision = tp / (tp + fp)
        return precision
    except ZeroDivisionError:
        print("Not enough input data to get precision")

def get_recall(tp, fn, fp, tn):
    try:
        recall = tp / (tp + fn)
        return recall
    except:
        print("Not enough input data to get recall")

def get_f1_score(tp, fn, fp, tn):
    try:
        precision = get_precision(tp, fn, fp, tn)
        recall = get_recall(tp, fn, fp, tn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    except TypeError:
        print("Not enough input data to get F1 score")
        return 0
    

def get_sensitivity(tp, fn, fp, tn):
    try:
        sensitivity = tp / (tp + fn)
        return sensitivity
    except ZeroDivisionError:
        print("Not enough input data to get sensitivity")
        return 0
        

def get_specificity(tp, fn, fp, tn):
    try:
        specificity = tn / (tn + fp)
        return specificity
    except:
        print("Not enough true negatives or false positives to get sensitivity")