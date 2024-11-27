import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, MixedSetVar, Problem
from mealpy.utils.agent import Agent
from mealpy.utils.problem import Problem
from mealpy.optimizer import Optimizer  

from mealpy.swarm_based.AO import OriginalAO
from mealpy.bio_based.SMA import OriginalSMA
from mealpy.human_based.HBO import OriginalHBO
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.BA import OriginalBA

import xgboost as xgb
from xgboost import XGBClassifier

from typing import Optional, List, Tuple, Union, Literal, Dict

np.random.seed(42)
random.seed(42)

# Custom Optimizer (Fix missing imports and minor logic errors)
class SGDOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, classifier_name='SGD', **kwargs):
        self.data = data
        self.classifier_name = classifier_name
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        penalty, alpha, l1_ratio, loss = x_decoded["penalty"], x_decoded["alpha"], x_decoded["l1_ratio"], x_decoded["loss"]
        sgd = SGDClassifier(
            penalty=penalty, 
            alpha=alpha, 
            l1_ratio=l1_ratio, 
            loss=loss, 
            random_state=42
        )
        
        # Use cross-validation to get a more robust performance estimate
        cv_scores = cross_val_score(
            sgd, 
            self.data["X_train"], 
            self.data["y_train"], 
            cv=5,  # 5-fold cross-validation
            scoring='accuracy'
        )
        
        return np.mean(cv_scores)

class BaggingOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        n_estimators_paras, max_features_paras, bootstrap_paras, bootstrap_features_paras= x_decoded["n_estimators"], x_decoded["max_features"], x_decoded["bootstrap"], x_decoded["bootstrap_features"]
        bagging = BaggingClassifier(
            n_estimators=n_estimators_paras, 
            max_features=max_features_paras, 
            bootstrap=bootstrap_paras, 
            bootstrap_features=bootstrap_features_paras,
            random_state=42
        )
        
        # Use cross-validation
        cv_scores = cross_val_score(
            bagging, 
            self.data["X_train"], 
            self.data["y_train"], 
            cv=5,
            scoring='accuracy'
        )
        
        return np.mean(cv_scores)

class DecisionTreeOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        min_samples_leaf, min_samples_split, max_features, criterion = x_decoded["min_samples_leaf"], x_decoded["min_samples_split"], x_decoded['max_features'], x_decoded['criterion']
        if max_features == "None":
            max_features = None
        decision_tree = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf, 
            min_samples_split=min_samples_split, 
            max_features=max_features, 
            criterion=criterion,
            random_state=42
        )
        
        # Use cross-validation
        cv_scores = cross_val_score(
            decision_tree, 
            self.data["X_train"], 
            self.data["y_train"], 
            cv=5,
            scoring='accuracy'
        )
        
        return np.mean(cv_scores)

class PerceptronOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        penalty, alpha, l1_ratio= x_decoded["penalty"], x_decoded["alpha"], x_decoded["l1_ratio"]
        perceptron = Perceptron(
            penalty=penalty, 
            alpha=alpha, 
            l1_ratio=l1_ratio,
            random_state=42
        )
        
        # Use cross-validation
        cv_scores = cross_val_score(
            perceptron, 
            self.data["X_train"], 
            self.data["y_train"], 
            cv=5,
            scoring='accuracy'
        )
        
        return np.mean(cv_scores)

class NearestCentroidOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        shrink_threshold, metric = x_decoded["shrink_threshold"], x_decoded["metric"],
        nearest = NearestCentroid(
            shrink_threshold=shrink_threshold,
            metric=metric
        )
        
        # Use cross-validation
        cv_scores = cross_val_score(
            nearest, 
            self.data["X_train"], 
            self.data["y_train"], 
            cv=5,
            scoring='accuracy'
        )
        
        return np.mean(cv_scores)

class XGBoostOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        n_estimators = x_decoded["n_estimators"]
        max_depth = x_decoded["max_depth"]
        learning_rate = x_decoded["learning_rate"]
        min_child_weight = x_decoded["min_child_weight"]
        subsample = x_decoded["subsample"]
        colsample_bytree = x_decoded["colsample_bytree"]
        
        xgb_classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        
        # Use cross-validation
        cv_scores = cross_val_score(
            xgb_classifier, 
            self.data["X_train"], 
            self.data["y_train"], 
            cv=5,
            scoring='accuracy'
        )
        
        return np.mean(cv_scores)

class CustomOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.progress_placeholder = st.empty()
        self.chart_placeholder_current = st.empty()
        self.chart_placeholder_global = st.empty()
        self.chart_placeholder_explo = st.empty()
        self.chart_placeholder_runtime = st.empty()
        self.chart_placeholder_diversity = st.empty()        
        self.fig_current = go.Figure()
        self.fig_global = go.Figure()
        self.fig_explo = go.Figure()
        self.fig_runtime = go.Figure()
        self.fig_diversity = go.Figure()
        self.explor = []
        self.exploi = []
        self.ddiversity = []
        
    def track_optimize_step(self, pop: List[Agent] = None, epoch: int = None, runtime: float = None) -> None:
        if self.problem.save_population:
            self.history.list_population.append(CustomOptimizer.duplicate_pop(pop))
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1].target.fitness)
        self.history.list_current_best_fit.append(self.history.list_current_best[-1].target.fitness)
        pos_matrix = np.array([agent.solution for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        self.ddiversity.append(np.mean(div, axis=0))
        
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        # # self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        # # self.history.list_exploitation = 100 - self.history.list_exploration
        self.explor.append(100 * (np.array(self.history.list_diversity[-1]) / div_max))
        self.exploi.append((100 - self.explor[-1]))
        self.plotting()
        
        self.logger.info(f">>>Problem: {self.problem.name}, Epoch: {epoch}, Current best: {self.history.list_current_best[-1].target.fitness}, "
                         f"Global best: {self.history.list_global_best[-1].target.fitness}, Runtime: {runtime:.5f} seconds")


    def plotting(self):
        # Plot for Current Best Fitness
        self.fig_current.data = []
        self.fig_current.add_trace(go.Scatter(
            y=self.history.list_current_best_fit,  
            mode='lines+markers',
            name="Current Best"
        ))
        self.fig_current.update_layout(
            title='Current Best Fitness Progress',
            xaxis_title='Iteration',
            yaxis_title='Fitness Score',
            # yaxis_range=[0, 1]
        )
        self.chart_placeholder_current.plotly_chart(self.fig_current, use_container_width=True)

        # Plot for Global Best Fitness
        self.fig_global.data = []
        self.fig_global.add_trace(go.Scatter(
            y=self.history.list_global_best_fit,  
            mode='lines+markers',
            name="Global Best"
        ))
        self.fig_global.update_layout(
            title='Global Best Fitness Progress',
            xaxis_title='Iteration',
            yaxis_title='Fitness Score',
            # yaxis_range=[0, 1]
        )
        self.chart_placeholder_global.plotly_chart(self.fig_global, use_container_width=True)

        # Plot for Exploitation and Exploration
        self.fig_explo.data = []
        self.fig_explo.add_trace(go.Scatter(
            y=self.explor, 
            mode='lines+markers',
            name="Exploration %"
        ))
        self.fig_explo.add_trace(go.Scatter(
            y=self.exploi, 
            mode='lines+markers',
            name="Exploitation %"
        ))
        self.fig_explo.update_layout(
            title='Exploration and Exploitation Progress',
            xaxis_title='Iteration',
            yaxis_title='Percentage',
            # yaxis_range=[-1, 101]
        )
        self.chart_placeholder_explo.plotly_chart(self.fig_explo, use_container_width=True)       

        # Plot for Diversity
        self.fig_diversity.data = []
        self.fig_diversity.add_trace(go.Scatter(
            y=self.ddiversity, 
            mode='lines+markers',
            # name="Exploration %"
        ))
        self.fig_diversity.update_layout(
            title='Diversity Measurement Chart',
            xaxis_title='Iteration',
            yaxis_title='Diversity Measurement',
            # yaxis_range=[0, 10]
        )
        self.chart_placeholder_diversity.plotly_chart(self.fig_diversity, use_container_width=True)    
       # Plot for Runtime
        self.fig_runtime.data = []
        self.fig_runtime.add_trace(go.Scatter(
            y=self.history.list_epoch_time,  
            mode='lines+markers',
            # name="Exploration %"
        ))
        self.fig_runtime.update_layout(
            title='Runtime Progress',
            xaxis_title='Iteration',
            yaxis_title='Second',
            # yaxis_range=[0, 1]
        )
        self.chart_placeholder_runtime.plotly_chart(self.fig_runtime, use_container_width=True)    
        
        # Display the current global best fitness as a progress text
        current_best = np.array(self.history.list_global_best[-1].target.fitness)
        self.progress_placeholder.text(f"Current best fitness: {current_best:.4f}")
    


class CustomOriginalAO(OriginalAO, CustomOptimizer):
    pass

class CustomOriginalHBO(OriginalHBO, CustomOptimizer):
    pass

class CustomOriginalSMA(OriginalSMA, CustomOptimizer):
    pass

class CustomOriginalGWO(OriginalGWO, CustomOptimizer):
    pass
class CustomOriginalBA(OriginalBA, CustomOptimizer):
    pass

def hyperparameter_tuning(data, algo_ml, algo_meta, epoch=100, pop_size=100, max_early_stop=None, mode="single", n_worker=None):
    if algo_ml == 'SGD':
        my_bounds = [
            FloatVar(lb=0.00000001, ub=10, name="alpha"),
            FloatVar(lb=0, ub=1, name="l1_ratio"),
            StringVar(valid_sets=('l1', 'l2', 'elasticnet'), name="penalty"),
            StringVar(valid_sets=('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'), name="loss")
        ]
        problem = SGDOptimizedProblem(bounds=my_bounds, minmax="max", data=data, verbose=True, save_population=True)
    if algo_ml == "Bagging Clasifier":
        my_bounds = [
                    FloatVar(lb=0.0001, ub=1, name="max_features"),
                    IntegerVar(lb=10, ub=1000, name='n_estimators'),
                    BoolVar(n_vars=1, name='bootstrap'),
                    BoolVar(n_vars=1, name='bootstrap_features'),
                    ]
        problem = BaggingOptimizedProblem(bounds=my_bounds, minmax="max", data=data,         
                                      verbose=True, save_population=True)
    if algo_ml == 'Decision Tree':
        my_bounds = [
                    IntegerVar(lb=1, ub=10, name="min_samples_leaf"),
                    IntegerVar(lb=2, ub=10, name="min_samples_split"),
                    StringVar(valid_sets=('sqrt', 'log2', 'None'), name="max_features"),
                    StringVar(valid_sets=('gini', 'entropy', 'log_loss'), name="criterion")
                    ]
        problem = DecisionTreeOptimizedProblem(bounds=my_bounds, minmax="max", data=data,         
                                      verbose=True, save_population=True)
    if algo_ml == 'Perceptron':
        my_bounds = [
                    FloatVar(lb=0.00000001, ub=10, name="alpha"),
                    FloatVar(lb=0, ub=1, name="l1_ratio"),
                    StringVar(valid_sets=('l1', 'l2', 'elasticnet'), name="penalty"),
                    ]
        problem = PerceptronOptimizedProblem(bounds=my_bounds, minmax="max", data=data,         
                                      verbose=True, save_population=True)      
    if algo_ml == 'Nearest Centroid':
        my_bounds = [
                    FloatVar(lb=0.00001, ub=10000, name="shrink_threshold"),
                    StringVar(valid_sets=('euclidean', 'manhattan'), name="metric"),
                    ]
        problem = NearestCentroidOptimizedProblem(bounds=my_bounds, minmax="max", data=data,         
                                      verbose=True, save_population=True)  
       
    if algo_ml == 'XGBoost':
        my_bounds = [
            IntegerVar(lb=10, ub=300, name="n_estimators"),
            IntegerVar(lb=1, ub=15, name="max_depth"),
            FloatVar(lb=0.01, ub=1, name="learning_rate"),
            IntegerVar(lb=1, ub=10, name="min_child_weight"),
            FloatVar(lb=0.5, ub=1, name="subsample"),
            FloatVar(lb=0.5, ub=1, name="colsample_bytree")
        ]
        problem = XGBoostOptimizedProblem(bounds=my_bounds, minmax="max", data=data,         
                                      verbose=True, save_population=True)

    if algo_meta == 'AO':
        model = CustomOriginalAO(epoch=epoch, pop_size=pop_size)
    if algo_meta == 'HBO':
        model = CustomOriginalHBO(epoch=epoch, pop_size=pop_size)
    if algo_meta == 'SMA':
        model = CustomOriginalSMA(epoch=epoch, pop_size=pop_size)
    if algo_meta == 'GWO':
        model = CustomOriginalGWO(epoch=epoch, pop_size=pop_size)
    if algo_meta == 'BA':
        model = CustomOriginalBA(epoch=epoch, pop_size=pop_size)

    term_dict = {
        "max_fe": 10000,
        "max_early_stop": max_early_stop
    }

    model.solve(problem, mode=mode, n_workers=n_worker, termination=term_dict, seed=10)
    print(f"Best accuracy: {model.g_best.target.fitness}")
    print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")
    return model


def main():
    st.set_page_config(
        page_title="ML Hyperparameter Tuning",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🤖 Machine Learning Hyperparameter Tuning")
    
    # Introduction
    st.markdown("""
    ### Welcome to the ML Hyperparameter Tuning Dashboard!

    This interactive tool helps you optimize machine learning model performance through advanced hyperparameter tuning techniques. 
    
    🔬 **How It Works:**
    1. Choose a dataset
    2. Define Features and Target
    3. Select machine learning and metaheuristic algorithms
    4. Customize hyperparameter tuning settings
    5. Compare default and optimized model performance
    """)
    
    st.sidebar.title("🛠️ Configuration Panel")
    
    st.sidebar.info("""
    **What is Hyperparameter Tuning?**
    - Hyperparameters are settings that aren't learned from the data
    - Tuning helps find the best configuration to improve model performance
    - We use metaheuristic algorithms to efficiently search the hyperparameter space
    """)
    
    dataset = st.sidebar.selectbox(
            "Choose a Dataset",
            options=["Alzheimer"],
            index=0,
            help="Select the dataset for model training and evaluation"
        )
    
    if dataset == 'Alzheimer':
        Data = pd.read_csv('data/alzheimers_disease_data.csv')
        # st.write("Preview of Dataset:")
        # st.dataframe(Data.head())
        
        # Dataset Preview Section
        with st.expander("📊 Dataset Preview", expanded=True):
            st.markdown("### Dataset Insights")
            st.write("First few rows of the Alzheimer's Disease dataset:")
            st.dataframe(Data.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", len(Data), help="Total number of data points in the dataset")
            with col2:
                st.metric("Total Features", len(Data.columns)-1, help="Number of feature columns in the dataset")
        
        st.sidebar.subheader("Data Settings")
        with st.sidebar.expander("Choose Features and Target", expanded=True):
            columns = Data.columns.tolist()
            feature_cols = st.multiselect(
                "Select Feature Columns", columns, default=columns[:-1]
            )
            target_col = st.selectbox(
                "Select Target Column", [""] + columns, index=len(columns)
            )
            
            if not feature_cols or not target_col:
                st.sidebar.warning("Please select features and target to proceed.")
                return

        # Advanced Data Settings
        with st.sidebar.expander("Advanced Data Settings", expanded=False):
            scaling_option = st.selectbox(
                "Choose Scaling Method",
                options=["None", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"],
                index=0
            )

        # Data Preprocessing
        X = Data[feature_cols]
        y = Data[target_col]
        
        # Scaling
        if scaling_option != "None":
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

            scalers = {
                "Standard Scaling": StandardScaler(),
                "Min-Max Scaling": MinMaxScaler(),
                "Robust Scaling": RobustScaler()
            }
            scaler = scalers[scaling_option]
            X = scaler.fit_transform(X)
            st.sidebar.success(f"{scaling_option} applied successfully!")

        # Train-Test Split
        from sklearn.model_selection import train_test_split
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test
        }

        # Sidebar for Algorithm Selection
        st.sidebar.header("Algorithm Configuration")
        algo_ml = st.sidebar.selectbox(
        "Choose a Machine Learning Algorithm",
        options=["SGD", "Perceptron", "Nearest Centroid", "Bagging Clasifier", "Decision Tree", "XGBoost"],
        index=0
        )
        algo_meta = st.sidebar.selectbox(
            "Choose a Metaheuristic Algorithm",
            options=["SMA", "HBO", "AO", 'GWO', "BA"],
            index=0
        )

        # Advanced Algorithm Settings
        with st.sidebar.expander("Advanced Algorithm Settings", expanded=False):
            epoch = st.number_input(
                "Number of Epochs", min_value=1, max_value=1000, value=3, step=1
            )
            pop_size = st.number_input(
                "Population Size", min_value=10, max_value=1000, value=100, step=10
            )
            max_early_stop = st.number_input(
                "Max Early Stop", min_value=1, max_value=100, value=10, step=1
            )
            mode = st.selectbox(
                "Optimization Mode",
                options=["single", "thread"],
                index=0
            )
            n_worker = st.number_input(
                "Number of Workers", min_value=1, max_value=10, value=1, step=1
            )

        def display_chart_descriptions(model):
            # Create a tabbed interface for different insights
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏆 Fitness Progress", 
                "🔍 Exploration vs Exploitation", 
                "⏱️ Runtime Analysis", 
                "🌈 Diversity Measurement", 
                "🔬 Best Hyperparameters"
            ])

            with tab1:
                st.markdown("## 🏆 Fitness Progress Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Current Best Fitness")
                    st.markdown("""
                    Tracks the best fitness score of the current population in each iteration.
                    - 📈 Shows how quickly the current population improves
                    - 🎯 Indicates the model's learning efficiency
                    - Higher values (closer to 1) represent better performance
                    """)

                with col2:
                    st.markdown("### Global Best Fitness")
                    st.markdown("""
                    Represents the overall best fitness score found across all iterations.
                    - 🌟 Tracks the global optimum discovered
                    - 🚀 Shows the algorithm's ability to find optimal solutions
                    - Convergence to a stable high value indicates successful optimization
                    """)

            with tab2:
                st.markdown("## 🔍 Exploration vs Exploitation Dynamics")
                st.markdown("""
                ### Understanding the Search Strategy
                - **Exploration (🔎)**: Searching new areas of the solution space
                - **Exploitation (🎯)**: Refining solutions around promising regions
                
                #### Key Insights:
                - Balance between exploration and exploitation is crucial
                - Early stages: More exploration (higher percentage)
                - Later stages: More exploitation (higher percentage)
                """)
                
                st.markdown("""
                ##### What This Means:
                - 🌍 Exploration helps prevent getting stuck in local optima
                - 🎳 Exploitation helps fine-tune the best solutions found
                - The changing percentages show the algorithm's adaptive search strategy
                """)

            with tab3:
                st.markdown("## ⏱️ Runtime Performance Analysis")
                st.markdown("""
                ### Computational Efficiency Insights
                - 🕒 Tracks the time taken for each optimization iteration
                - 🚀 Helps understand computational complexity
                
                #### What to Look For:
                - Consistent runtime suggests stable algorithm performance
                - Sudden spikes might indicate complex solution spaces
                - Helps in comparing different metaheuristic algorithms
                """)

            with tab4:
                st.markdown("## 🌈 Diversity Measurement")
                st.markdown("""
                ### Population Diversity Tracking
                - 🧬 Measures the spread of solutions in the search space
                - 🔬 Indicates how different the solutions are from each other
                
                #### Diversity Insights:
                - High diversity: Exploring many different solution regions
                - Low diversity: Focusing on a specific solution area
                - Crucial for avoiding premature convergence
                """)

            with tab5:
                st.markdown("## 🔬 Best Hyperparameters Breakdown")
                
                # Retrieve and display best parameters
                best_params = model.problem.decode_solution(model.g_best.solution)
                
                st.markdown("### Optimal Configuration Found")
                
                # Create an expandable section for each parameter
                for param, value in best_params.items():
                    with st.expander(f"🔑 {param.replace('_', ' ').title()}"):
                        st.write(f"**Value:** {value}")
                        
                        # Add context-specific descriptions
                        param_descriptions = {
                            "Alpha": "Regularization strength. Controls model complexity and prevents overfitting.",
                            "Loss": "The loss function determines how the model penalizes prediction errors.",
                            "Penalty": "Type of regularization applied to prevent model from becoming too complex.",
                            "L1_Ratio": "Balance between L1 and L2 regularization (useful for feature selection).",
                            "N_Estimators": "Number of trees or iterations in ensemble methods.",
                            "Max_Depth": "Maximum depth of decision trees, controlling model complexity.",
                            "Learning_Rate": "Step size at each iteration while moving toward a minimum of a loss function.",
                            "Subsample": "Fraction of samples used for fitting individual trees.",
                            "Colsample_Bytree": "Fraction of features used for each tree construction.",
                            "Max_Features": "Maximum number of features considered when splitting a node (e.g., 'sqrt', 'log2').",
                            "Min_Samples_Leaf": "Minimum number of samples required to be at a leaf node.",
                            "Min_Samples_Split": "Minimum number of samples required to split an internal node.",
                            "Criterion": "Function to measure the quality of a split (e.g., 'gini', 'entropy', 'log_loss').",
                            "Bootstrap": "Whether bootstrap sampling is used when building trees.",
                            "Bootstrap Features": "Whether features are sampled with replacement when building trees.",
                            "Shrink_Threshold": "Threshold for shrinking centroids in Nearest Centroid classification.",
                            "Metric": "Distance metric used for Nearest Centroid classification (e.g., 'euclidean', 'manhattan').",
                            "Min_Child_Weight": "Minimum sum of instance weights (hessian) needed in a child node (XGBoost)."
                        }
                        
                        # Try to find and display a description
                        description = next((desc for key, desc in param_descriptions.items() if key.lower() in param.lower()), "No additional context available.")
                        st.markdown(f"**Context:** {description}")

            return best_params
        
        # Start Button
        if st.sidebar.button("Start Hyperparameter Tuning"):
            # Default Parameters Performance Section
            st.markdown("""
            ### 🔍 Default Parameters Performance
            Before hyperparameter tuning, we evaluate the model's performance using default settings.
            This helps us understand the baseline performance and see how much improvement we can achieve.
            """)
            
            # Select and train the default model based on the chosen algorithm
            if algo_ml == 'SGD':
                default_model = SGDClassifier(random_state=42)
            elif algo_ml == 'Perceptron':
                default_model = Perceptron(random_state=42)
            elif algo_ml == 'Decision Tree':
                default_model = DecisionTreeClassifier(random_state=42)
            elif algo_ml == 'Bagging Clasifier':
                default_model = BaggingClassifier(random_state=42)
            elif algo_ml == 'Nearest Centroid':
                default_model = NearestCentroid()
            elif algo_ml == 'XGBoost':
                default_model = XGBClassifier(random_state=42)
            
            # Fit and predict with default parameters
            default_model.fit(data["X_train"], data["y_train"])
            y_pred_default = default_model.predict(data["X_test"])
            
            # Default Model Evaluation
            # Classification Report Section
            st.markdown("""
            #### 📄 Classification Report
            A comprehensive report showing precision, recall, and F1-score for each class.
            - **Precision**: Accuracy of positive predictions
            - **Recall**: Proportion of actual positives correctly identified
            - **F1-Score**: Harmonic mean of precision and recall
            """)
            default_report = classification_report(y_test, y_pred_default, output_dict=True)
            default_report_df = pd.DataFrame(default_report).transpose()
            st.dataframe(default_report_df.style.format(precision=2))


            # Confusion Matrix Section
            st.markdown("""
            #### 📊 Confusion Matrix
            Visualizes the model's prediction performance:
            - Diagonal values show correct predictions
            - Off-diagonal values indicate misclassifications
            """)
            default_conf_matrix = confusion_matrix(y_test, y_pred_default)
            fig_default, ax_default = plt.subplots(figsize=(6, 4))
            sns.heatmap(default_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
            ax_default.set_title("Confusion Matrix - Default Parameters")
            ax_default.set_xlabel("Predicted Labels")
            ax_default.set_ylabel("True Labels")
            st.pyplot(fig_default)

            st.markdown(f"#### Default Parameters - Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
            
            # Horizontal line to separate default and tuned results
            st.markdown("---")
            
            # HYPERPARAMETER TUNING START
            st.markdown(f"### Tuning Hyperparameter with `{algo_ml}` and `{algo_meta}`")
            
            st.sidebar.markdown("### 🧠 Metaheuristic Algorithms")
            st.sidebar.info("""
            Metaheuristic algorithms mimic natural phenomena to find optimal solutions:
            - **SMA**: Slime Mould Algorithm (nature-inspired)
            - **HBO**: Hunger-Based Optimization
            - **AO**: Arithmetic Optimization Algorithm
            - **GWO**: Grey Wolf Optimizer
            - **BA**: Bat Algorithm
            """)

            # Hyperparameter tuning logic here (reusing previous implementation)
                    # Hyperparameter tuning
            model = hyperparameter_tuning(
                data=data, algo_ml=algo_ml, algo_meta=algo_meta,
                epoch=epoch, pop_size=pop_size, max_early_stop=max_early_stop,
                mode=mode, n_worker=n_worker
            )
            
            # Display chart descriptions and additional insights
            best_params = display_chart_descriptions(model)
             
            # Horizontal line to separate default and tuned results
            st.markdown("---")

            # Decode parameters and train the final model
            param = model.problem.decode_solution(model.g_best.solution)
            if algo_ml == 'SGD':
                model_fix = SGDClassifier(loss=param['loss'], alpha=param['alpha'], l1_ratio=param['l1_ratio'], penalty=param['penalty'], random_state=42)
            if algo_ml == 'Perceptron':
                model_fix = Perceptron()
            if algo_ml == 'Decision Tree':
                model_fix = DecisionTreeClassifier()
            if algo_ml =='Bagging Clasifier':
                model_fix = BaggingClassifier()
            if algo_ml == 'Nearest Centroid':
                model_fix = NearestCentroid()
            if algo_ml == 'XGBoost':
                model_fix = XGBClassifier(
                    n_estimators=param['n_estimators'],
                    max_depth=param['max_depth'],
                    learning_rate=param['learning_rate'],
                    min_child_weight=param['min_child_weight'],
                    subsample=param['subsample'],
                    colsample_bytree=param['colsample_bytree'],
                    random_state=42
                )
                
            model_fix.fit(data["X_train"], data["y_train"])
            y_pred = model_fix.predict(data["X_test"])

            # Model Evaluation Section
            st.subheader("Model Evaluation")
            st.markdown("#### Best Hyperparameters")
            st.write(param)

            st.markdown("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(precision=2))

            st.markdown("#### Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)

            st.markdown(f"#### Accuracy: {accuracy_score(y_test, y_pred):.4f}")

            st.toast("Hyperparameter tuning completed!", icon='🎉')
            st.success("Hyperparameter tuning completed!", icon='🎉')
        else:
            st.markdown("### Waiting for Hyperparameter Tuning to Start")
            st.write("Click **Start Hyperparameter Tuning**.")
             


if __name__ == "__main__":
    main()
