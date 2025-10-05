import base64
import time
import streamlit as st
import joblib
import pandas as pd
import os.path

from preprocess_data import preprocess_dataset

from model import run_inference
from metrics import model_evaluation, visualize_confusion_matrix

@st.cache_data
def convert_for_download(df):
    return pd.DataFrame(df).to_csv(index=False).encode("utf-8")

def main():    
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

    st.set_page_config(page_title="ExoSeeker", page_icon=":material/planet:",layout="wide")

    st.title(":blue[:material/planet:] ExoSeeker")
    
    train_col, model_col, predict_col = st.columns(3, border=True)
    
    # Added here to be accessed later for model evaluation
    model = None
    y_test = None
    y_pred = None
    
    with train_col:
        with st.container(height=400, border=False):
            st.subheader(":material/dataset: Kelper Exoplanet Data", help="""The uploaded CSV file here must be the [Kepler Objects of Interest Dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative). The data will be preprocessed once the Train New Model has been pressed. The data preprocessing for model fitting involves dropping unnecessary columns, such as assigned exoplanet names, and retaining important features, including transit properties.\n \nMissing values are filled with the average value of the entire column. The exoplanet disposition is split and used as the target column. A standard scaler is used to remove the mean and achieve unit variance, or a standard dispersion of values, thereby reducing the impact of outliers. The dataset is then split for training and testing the model or estimator, and then the evaluation of the performance of the model is shown in Model Evaluation automatically.""")
            uploaded_file = st.file_uploader(":material/data_table: Upload Dataset for Model Fitting", type=["csv"], accept_multiple_files=False, key="uploaded_training_dataset_file")
            if uploaded_file is not None:
                uploaded_dataset = pd.read_csv(uploaded_file)
                
                #st.write(uploaded_dataset)
                st.success("Training dataset uploaded successfully")
                
    with model_col:
        # Default models" hyperparaameters
        input_rf_n_estimators=200
        input_rf_max_depth=1
        input_gb_n_estimators=100
        input_gb_max_depth=3
        input_mlp_max_iter=200
        input_mlp_alpha=0.0001
            
        with st.container(height=400, border=False):
            st.subheader(":material/graph_7: Model Build", help="""In Model Build, a stacking classifier is utilized, wherein outputs from multiple estimators can be stacked and a final classifier is used to compute the final prediction, in this case, logistic regression. Stacking allows the use of the strength of each estimator by using their output as input to a final estimator. You can choose to mix and match different estimators and tweak their main hyperparameters.""")
            
            estimator_options = ["Random Forest", "Gradient Boosting", "Multi-layer Perceptron"]
            choosen_classifiers = st.pills(":material/graph_4: Select Estimators", estimator_options, selection_mode="multi", default="Random Forest")
            
            st.write(":material/settings_input_component: Main Hyperparameters")
            

            if "Random Forest" in choosen_classifiers:
                with st.expander("Random Forest"):
                
                    input_rf_n_estimators = st.slider("Random Forest N Estimators", min_value=50, max_value=1000, value=100, step=50, key="input_rf_n_estimators", help="The number of trees in the forest.")
                    
                    input_rf_max_depth = st.slider("Random Forest Max Depth", min_value=0, max_value=10, value=3, step=1, key="input_rf_max_depth", help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
            else:
                pass
                    
            if "Gradient Boosting" in choosen_classifiers:
                with st.expander("Gradient Boosting"):
                
                    input_gb_n_estimators = st.slider("Gradient Boosting N Estimators", min_value=50, max_value=1000, value=100, step=50, key="input_gb_n_estimators", help="The number of boosting stages to perform.")
                    
                    input_gb_max_depth = st.slider("Gradient Boosting Max Depth", min_value=0, max_value=10, value=3, step=1, key="input_gb_max_depth", help="Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.")
            else:
                pass
                    
            if "Multi-layer Perceptron" in choosen_classifiers:
                with st.expander("Multi-layer Perceptron"):
                    input_mlp_max_iter = st.slider("Multi-layer Perceptron Max Iterations", min_value=50, max_value=1000, value=100, step=50, key="input_mlp_max_iter", help="Maximum number of iterations.")
                    
                    input_mlp_alpha = st.number_input("Multi-layer Perceptron Alpha", value=0.0001, placeholder=0.0001, key="input_mlp_alpha", help="Strength of the L2 regularization term. The L2 regularization term is divided by the sample size when added to the loss. Essentially, the magnitude of the change in the weights of the neural network. The default alpha is 0.0001.")
            else:
                pass
        
            if st.button("Train New Model", type="primary", icon=":material/rocket_launch:"):
                try:
                    with st.spinner("Training model...", show_time=True):
                        st.toast("Preprocessing data...")
                        st.toast("Dropping unessential column features...")
                        st.toast("Filling missing values with average value...")
                        X_train, X_test, y_train, y_test = preprocess_dataset(uploaded_dataset, target=False)
                            
                        st.toast("Fitting the estimators...")
                            
                        model = run_inference(X_train=X_train, y_train=y_train, choosen_classifiers=choosen_classifiers, rf_n_estimators=input_rf_n_estimators, rf_max_depth=input_rf_max_depth, gb_n_estimators=input_gb_n_estimators, gb_max_depth=input_gb_max_depth, mlp_max_iter=input_mlp_max_iter, mlp_alpha=input_mlp_alpha)
                            
                        st.toast("Running predictions for model evaluation...")
                        
                        y_pred = model.predict(X_test)
                            
                        st.toast("Saving model on local machineas model.pkl...")
                            
                        joblib.dump(model, "model.pkl")
                            
                        st.success("The model has been trained successfully!")
                        
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
                
    with predict_col:
        with st.container(height=400, border=False):
            st.subheader(":material/area_chart: Target Data Forecast", help="A cumulative KOI dataset without the target variable, the exoplanet disposition, can be uploaded here for the fitted model to make predictions upon it. Once the model has finished, you can download the forecasted classification of each potential exoplanet. Each row would be classified as either a candidate or a confirmed exoplanet.")
            uploaded_target = st.file_uploader(":material/data_table: Upload Target Data", type=["csv"], accept_multiple_files=False, key="uploaded_target_data")
            if uploaded_target is not None:
                target_data = pd.read_csv(uploaded_target)
                
            if st.button("Predict", type="primary", icon=":material/arrow_shape_up_stack:"):
                try:
                    X_target = preprocess_dataset(target_data, target=True)
                    
                    if os.path.exists("model.pkl"):
                        model = joblib.load("model.pkl")
                        y_pred_target = model.predict(X_target)
                        
                        csv = convert_for_download(y_pred_target)
                        
                    else:
                        st.write(f"No classifiers detected, please train a new one!")
                    

                    st.download_button(
                        label="Download Raw Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )

                except Exception as e:
                    st.error(f"Error in prediction: {e}")
                

    model_eval_col = st.container(border=True, height="content")
      
    with model_eval_col:
        st.subheader(":material/analytics: Model Evaluation", help="""The built model's performance is measured in: accuracy, the percentage of data predicted correctly; sensitivity, the percentage of data predicted correctly that belongs to positive classes; specificity, percentage of data predicted correctly that belongs to negative classes; Precision, percentage of correct positive predictions made; and the F1 score, a performance metric derived from the precision and recall of the model.

A confusion matrix is also generated with the following values: true positives, exoplanets confirmed and classified as confirmed; true negatives, exoplanet candidates and classified as candidates; false positive, exoplanet candidates but classified as confirmed; and false negatives, exoplanets confirmed but classified as candidates. These values were obtained by comparing the predicted exoplanet disposition to the actual values in the test dataset.""")
        
        bar_col, heatmap_col = st.columns(2)
            
        if y_test is not None and y_pred is not None:
            with bar_col:
                model_evaluation(y_test, y_pred)
            
            with heatmap_col:     
                visualize_confusion_matrix(y_test, y_pred)
                
    st.markdown("""<center><p style="color:gray"><i>Ad astra!</i></p></center>""", unsafe_allow_html=True)
                
    
        
if __name__ == "__main__":
    main()