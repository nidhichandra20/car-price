import numpy as np
import pandas as pd
import streamlit as st
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types(1).csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()



st.title("GLASS TYPE PREDICTOR")
st.sidebar.title("Data Analysis")
if (st.sidebar.checkbox("Show raw data")):
    st.subheader("Full data set")
    st.dataframe(glass_df)


st.sidebar.subheader("Scatter plot")
feature_list = st.sidebar.multiselect("Select value", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in feature_list:
    st.subheader(f"Scatter Plot betwwen {i} and Glass Type")
    plt.figure(figsize = (15,5))
    plt.scatter(glass_df[i], glass_df["GlassType"])
    st.pyplot()

st.sidebar.subheader("Visualisation Select")
plot_types = st.sidebar.multiselect("Select charts" ,("Histogram", "Box plot", "Count plot", "Pie chart", "Correlation heatmap", "Line chart", "Area chart"))


if "Histogram" in plot_types:
    st.subheader("Histogram")
    fl = st.sidebar.selectbox("Select value for Histogram",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') )
    plt.figure(figsize=(15,5))
    plt.hist(glass_df[fl], bins = "sturges", edgecolor = "black")
    st.pyplot()


if "Box plot" in plot_types:
    st.subheader("Box plot")
    f_l = st.sidebar.selectbox("Select value for Box plot", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    plt.figure(figsize=(15,5))
    sns.boxplot(glass_df[f_l])
    st.pyplot()

if "Area chart" in plot_types:
    st.subheader("Area Chart")
    st.area_chart(glass_df)

if "Line chart" in plot_types:
    st.subheader("Line Chart")
    st.line_chart(glass_df)

if "Count plot" in plot_types:
    st.subheader("Count Plot")
    plt.figure(figsize = (15,5))
    sns.countplot(glass_df["GlassType"])
    st.pyplot()

if "Correlation heatmap" in plot_types:
    st.subheader("Correlation Heatmap")
    plt.figure(figsize = (15,5))
    sns.heatmap(glass_df.corr(), annot = True)
    st.pyplot()

if "Pie chart" in plot_types:
    st.subheader("Pie Chart")
    plt.figure(figsize = (15,5))
    data = glass_df["GlassType"].value_counts()
    plt.pie(data, labels = data.index)
    st.pyplot()

st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

from sklearn.metrics import plot_confusion_matrix

# if classifier =='Support Vector Machine', ask user to input the values of 'C','kernel' and 'gamma'.
if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()





if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimator = st.sidebar.number_input("Number of Trees", 100, 500, step = 5)
    max_depth_1 = st.sidebar.number_input("Depth of the Trees", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rfb_model=RandomForestClassifier(n_estimators = n_estimator, max_depth = max_depth_1, n_jobs = -1)
        rfb_model.fit(X_train,y_train)
        y_pred = rfb_model.predict(X_test)
        accuracy = rfb_model.score(X_test, y_test)
        glass_type = prediction(rfb_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rfb_model, X_test, y_test)
        st.pyplot()



if classifier == 'Logistic Regression':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C", 1, 100, step = 1)
    iteration = st.sidebar.number_input("Max iteration", 10, 1000, step = 10)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        lr_model=LogisticRegression(C = c_value, max_iter = iteration)
        lr_model.fit(X_train,y_train)
        y_pred = lr_model.predict(X_test)
        accuracy = lr_model.score(X_test, y_test)
        glass_type = prediction(lr_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(lr_model, X_test, y_test)
        st.pyplot()
