import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Experiment Results using One-class SVM and Gaussian Mixture Model \
    from Scikit-learn") 

select_slide = st.sidebar.selectbox(
    "Which slide would you like to navigate to?",
    ("One-class SVM", "Gaussian Mixture Model", "Results")
)

if select_slide == "One-class SVM":
    st.subheader("Method 1: One-class SVM") 

    st.write("Support Vector Machines can create a non-linear \
        decision boundary to seperate the data points.")
    st.write("By just providing the normal data, One-class SVM (OC-SVM) \
        can generate a function that returns positive scores for most \
        training data points. Data points with negative scores returned will \
        be judged as anomalies.")
    st.write("Take the negative value of the mean of data point scores as the \
        anomaly score for each audio file.")

    st.subheader(" ")
    st.write("Train the models using the parameter grid below:")
    ocsvm_parameter_grid_df = pd.read_csv('OC-SVM parameter grid.csv')
    st.table(ocsvm_parameter_grid_df)

elif select_slide == "Gaussian Mixture Model":
    st.subheader("Method 2: Gaussian Mixture Model") 

    st.write("A Gaussian mixture model assumes all the data points are \
        generated from a mixture of a finite number of Gaussian distributions \
        with unknown parameters.")
    st.write("This model intends to group data points into clusters that \
        indicate machines' section and attribute information.")
    st.write("Take the negative value of the maximum log-likihood as the \
        anomaly score for each data point. Then, take the mean value as the \
        anomaly score for each audio file.")

    st.subheader(" ")
    st.write("Train the models using the parameter grid below:")
    gmm_parameter_grid_df = pd.read_csv('GMM parameter grid.csv')
    st.table(gmm_parameter_grid_df)
    
elif select_slide == "Results":
    st.subheader("Best Results for OC-SVM and GMM") 
    result_df = pd.read_csv('Results.csv')
    col_ref = {'Trial 1:': 'background-color: #ffec8c', 
            'OC-SVM': 'background-color: #ffec8c', 
            'Trial 2:':'background-color: #c2f5ff',
            'GMM':'background-color: #c2f5ff'}
    st.table(result_df.style.apply(lambda x: pd.DataFrame(col_ref, \
        index=result_df.index, columns=result_df.columns).fillna(''), axis=None))
    
    col1, col2= st.columns(2)
    
    with col1:
        st.subheader("Best Parameters for OC-SVM:")
        st.write("kernel: rbf")
        st.write("nu (contamination): 0.0001")
    
    with col2:
        st.subheader("Best Parameters for GMM:")
        st.write("n_components: 40")
        st.write("covariance_type: 'diag'")
        st.write("max_iter: 150")
        st.write("init_params: 'kmeans'")


