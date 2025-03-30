# =============================================================================
# Import necessary libraries
# =============================================================================
import pandas as pd
import numpy as np
import os
import ast
import sys
from sklearn.ensemble import RandomForestClassifier
from TSC.time_series import generate
from TSC.classification.naive_based.naive import NaiveClassifier
from TSC.classification.shapelet_based.LearningShapeletClassifier import LearningShapeletClassifier
from TSC.classification.shapelet_based.ShapeletTransformClassifier import ShapeletTransformClassifier
from TSC.classification.interval_based.RandomIntervalSpectralEnsembleClassifier import RandomIntervalSpectralEnsembleClassifier
from TSC.classification.interval_based.TimeSeriesForestClassifier import TimeSeriesForestClassifier
from TSC.classification.hybrid.hivecotev2 import HIVECOTEV2
from TSC.classification.hybrid.RISTClassifier import RISTClassifier
from TSC.classification.feature_based.Catch22Classifier import Catch22Classifier
from TSC.classification.feature_based.TSFreshClassifier import TSFreshClassifier
from TSC.classification.distance_based.ElasticEnsemble import ElasticEnsemble
from TSC.classification.distance_based.KNeighborsTimeSeriesClassifier import KNeighborsTimeSeriesClassifier
from TSC.classification.dictionary_based.BOSSEnsemble import BOSSEnsemble
from TSC.classification.dictionary_based.WEASEL import WEASEL
from TSC.classification.deep_learning.CNNClassifier import CNNClassifier
from TSC.classification.deep_learning.FCNClassifier import FCNClassifier
from TSC.classification.deep_learning.InceptionTimeClassifier import InceptionTimeClassifier
from TSC.classification.convolution_based.RocketClassifier import RocketClassifier
from TSC.classification.cross_validation import TimeSeriesCrossValidator
from datetime import datetime
from TSC.metrics import results


# =============================================================================
# Set working directory
# =============================================================================
base_working_directory = '/lu/topola/home/mariasad/MS'
os.chdir(base_working_directory)
data_file_path = os.path.join(base_working_directory, 'data_noise.xlsx')

# =============================================================================
# Load Excel files
# =============================================================================
data = pd.read_excel(data_file_path)
g_number = int(sys.argv[1])
data = data[data['g_number'] == g_number]

# =============================================================================
# Define and create necessary directories
# =============================================================================
g_date = data['g_date'].iloc[0]

# Define working directory based on g_date and g_number
working_directory = os.path.join(base_working_directory, f"{g_date}_{g_number}") 

input_dir = os.path.join(working_directory, 'input_data')
output_dir = os.path.join(working_directory, 'output_data')
log_dir = os.path.join(working_directory, 'logi')
trained_models_dir = os.path.join(working_directory, 'trained_models')

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# Ensure all directories exist
for directory in [input_dir, output_dir, log_dir, trained_models_dir]:
    ensure_directory(directory)

# =============================================================================
# Logging configuration
# =============================================================================
log_filename = f'cross_validation_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'

# =============================================================================
# Generating time_series data if not exists
# =============================================================================
# Get the indexes and exists values for the matched date
random_seeds = eval(data.iloc[0]['d_random_seeds'])   # Extract seeds as a list
g_exists=int(data['g_exists'].iloc[0]) 

if g_exists == 0:
        
    # Now you can extract the required parameters from the current_row
    output_type = str(data['d_output_type'].iloc[0])
    shapes = ast.literal_eval(data['d_shapes'].iloc[0])
    length = int(data['d_length'].iloc[0])
    seed = random_seeds[0]  # You can adjust seed handling as needed
        
    noise_distribution = data['d_noise_distribution'].iloc[0]
    noise_max = data['d_noise_max'].iloc[0]
    parameters_list = eval(data['d_parameters_list'].iloc[0])  # Convert string to list
        
    # Apply noise dynamically if specified
    if pd.notna(noise_distribution) and pd.notna(noise_max):
        noise_max = float(noise_max)
        noise_distribution = str(noise_distribution)

        # Ensure the noise params are set based on the distribution
        if noise_distribution == 'norm':
            noise_params = {'noise': ('norm', 0, noise_max)}
        elif noise_distribution == 'uniform':
            noise_params = {'noise': ('uniform', -noise_max, noise_max)}
        else:
            noise_params = {}

        # Append the noise dictionary as a separate element in the parameters_list
        for i, shape_params in enumerate(parameters_list):
            # Ensure noise is added as a separate dictionary, not inside the shape params
            if isinstance(shape_params[-1], dict) and 'noise' not in shape_params[-1]:
                # If last element is the shape params, append noise as a new dictionary
                shape_params.append(noise_params)
            elif isinstance(shape_params[-1], dict) and 'noise' in shape_params[-1]:
                # Do nothing if noise is already present
                continue
            else:
                # Add noise as a separate dictionary if last element isn't a dict
                shape_params.append(noise_params)

    # Generate the time series data
    time_series, desc = generate.generate_time_series(output_type, shapes, parameters_list, length, seed)

    # Save time_series to a CSV file in ./input_data
    time_series_path = os.path.join(input_dir, 'time_series.csv')
    if isinstance(time_series, pd.DataFrame):
        # If time_series is a DataFrame, save it directly
        time_series.to_csv(time_series_path, index=False)
    elif isinstance(time_series, np.ndarray):
        # If time_series is a numpy array, convert it to DataFrame before saving
        pd.DataFrame(time_series).to_csv(time_series_path, index=False)

    # Save desc to a CSV file in ./input_data
    desc_path = os.path.join(input_dir, 'desc.csv')
    desc.to_csv(desc_path, index=False)
        
# =============================================================================
# Extract cross-validation parameters from the Excel file
# =============================================================================
cv_method = str(data.iloc[0]['d_cv_method'])  # Extract cv_method
n_splits = int(data.iloc[0]['d_n_splits']) if cv_method == 'StratifiedKFold' else None  # Handle n_splits if StratifiedKFold


# =============================================================================
# Helper function to check if a model should be built
# =============================================================================
def check_model_exists(model_col_name):
    return pd.notna(data.iloc[0][model_col_name])  # Return True if data exists in the model column

# =============================================================================
# Podział modeli na procesy
# =============================================================================
# Pobierz nazwy modeli z kolumn Excela
model_columns = [col for col in data.columns if col.startswith('m_')]
models = [col for col in model_columns if check_model_exists(col)]

# =============================================================================
# Dynamically create and evaluate models based on Excel file columns
# =============================================================================
models_to_evaluate = {
    "m_NaiveClassifier": NaiveClassifier,
    "m_LearningShapeletClassifier": LearningShapeletClassifier,
    "m_ShapeletTransformClassifier": ShapeletTransformClassifier,
    "m_RandomIntervalSpectralEnsembleClassifier": RandomIntervalSpectralEnsembleClassifier,
    "m_TimeSeriesForestClassifier": TimeSeriesForestClassifier,
    "m_HIVECOTEV2": HIVECOTEV2,
    "m_RISTClassifier": RISTClassifier,
    "m_Catch22Classifier": Catch22Classifier,
    "m_TSFreshClassifier": TSFreshClassifier,
    "m_ElasticEnsemble": ElasticEnsemble,
    "m_KNeighborsTimeSeriesClassifier": KNeighborsTimeSeriesClassifier,
    "m_BOSSEnsemble": BOSSEnsemble,
    "m_WEASEL": WEASEL,
    "m_CNNClassifier": CNNClassifier,
    "m_FCNClassifier": FCNClassifier,
    "m_InceptionTimeClassifier": InceptionTimeClassifier,
    "m_RocketClassifier": RocketClassifier    
}

for model_name, model_class in models_to_evaluate.items():
    
    # Get the value from the column corresponding to the model
    model_value = data.iloc[0][model_name]

    # Check if the model column exists and has data
    if check_model_exists(model_name):
            # Special case for NaiveClassifier that requires 'desc'
            if model_name == "m_NaiveClassifier":
                model = NaiveClassifier(desc)
            else:
                # Create the model instance for other models
                model = model_class()
                
            if model_name in ["m_Catch22Classifier", "m_TSFreshClassifier"]:
                # Use ast.literal_eval to safely evaluate the string
                param_grid = ast.literal_eval(data.iloc[0][model_name])
                
                # Check and replace estimator integers with RandomForestClassifier instances
                if "estimator" in param_grid:
                    param_grid["estimator"] = [
                        RandomForestClassifier(n_estimators=n) if isinstance(n, int) else n
                        for n in param_grid["estimator"]
                    ]
            else:
                param_grid = ast.literal_eval(data.iloc[0][model_name])

                
                
            # Set up TimeSeriesCrossValidator
            if cv_method == "LeaveOneOut":
                ts_cv = TimeSeriesCrossValidator(
                    model=model,
                    param_grid=param_grid,
                    time_series=time_series,
                    description=desc,
                    cv_method=cv_method,
                    random_seeds=random_seeds,
                    log_dir=log_dir,
                    log_filename=log_filename,
                    dir_output=output_dir
                )
            else:
                ts_cv = TimeSeriesCrossValidator(
                    model=model,
                    param_grid=param_grid,
                    time_series=time_series,
                    description=desc,
                    cv_method="StratifiedKFold",
                    n_splits=10,
                    shuffle=True,
                    random_seeds=random_seeds,
                    log_dir=log_dir,
                    log_filename=log_filename,
                    dir_output=output_dir
                )
        
            # Perform cross-validation
            all_params, train_metrics_df, test_metrics_df = ts_cv.cross_validate()
        
            
            
            
            
            
            
            









