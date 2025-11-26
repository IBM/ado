from autogluon.tabular import TabularPredictor
import pandas as pd
from autoconf.rule_based_classifier import is_row_valid
from autoconf.pydantic_models import JobConfig

import logging

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)

logger = logging.getLogger(__name__)
VALID_N_GPUS = [1,2,4,8]


def get_model_prediction_and_metadata(config: pd.DataFrame | dict | JobConfig, predictor )-> tuple[int|None , dict[str]]:
    """Gets valid/invalid prediction and reason why"""
    if isinstance(config, dict):
        config = pd.DataFrame(config, index=[0])
    if isinstance(config, JobConfig):
        config = pd.DataFrame([config.model_dump()], index=[0] )


    metadata = {}
    pred = None
    C_err = None
    b, RBC_err = is_row_valid(config, raise_error=False)
    if int(b) == 1:
        try:
            pred =  predictor.predict(config).values[0]
            logger.debug('Prediction succeeded')
        except Exception as e:
            logger.debug('Prediction FAILED')
            C_err = str(e)
        
    
    metadata['Rule-Based Classifier error'] = " ".join(RBC_err)
    metadata['Predictive Model Classifier error'] = C_err
    pred  = int(pred) if pred else pred
    return pred, metadata

class MinGpuRecommender():
    def __init__(self, job_config: JobConfig, predictor, valid_n_gpu :list[int] = VALID_N_GPUS):
        self.job_config = job_config
        self.predictor = predictor
        self.valid_n_gpu = valid_n_gpu

    def recommend_min_gpu(self):
        return recommend_min_gpu(self.job_config, self.predictor, self.valid_n_gpu)
    

def recommend_min_gpu(job_config: JobConfig, predictor, valid_n_gpu :list[int] = VALID_N_GPUS)->  tuple[int , dict[str]]:
    """ Recommends the minimum number of GPUs required for a SFT job defined by the fields of the pydantic model :job_config:
    Returns
        min_n_gpu: the minimum number of valid gpus
        -1 if no gpu number in the valid_n_gpu list is predicted to be valid"""
    res_dict = {}
    if isinstance(predictor, str):
        predictor = TabularPredictor.load(predictor)
        
    metadata_user_provided_config = 'User config was not provided'
    for n in valid_n_gpu:
        logger.info(f'Testing number_gpus={n}')
        if n == job_config.number_gpus:
            logger.info('This is the value provided by the user, for this configuration the recommender will provide additional metadata')
            p,m = get_model_prediction_and_metadata(job_config, predictor=predictor)
            res_dict[n] = p
            metadata_user_provided_config = m 
        else:
            new_job_config = job_config.model_copy(update={"number_gpus": n})
            p,m = get_model_prediction_and_metadata(new_job_config, predictor=predictor)
            res_dict[n] = p
        
        logger.info(f"Prediction:{p}\t(note:0 is not valid, 1 is Valid)")

    logger.info(f"Metadata related to the user provided config (number_gpus={job_config.number_gpus}):{metadata_user_provided_config}")
    

    min_key = min((k for k, v in res_dict.items() if int(v) == 1), default=-1)
    if min_key == -1:
        logger.info(f"I cannot provide a recommended number_gpus because no number of gpus in the list {valid_n_gpu} would result in a valid run according to the predictive model.")
    else:
        logger.info(f"The recommended number_gpus={min_key}.")

    return min_key, metadata_user_provided_config


def validate_as_jobconfig(config_to_test):
    from pydantic import ValidationError
    try:
        job = JobConfig(**config_to_test)
        print("Validation successful:", job)
    except ValidationError as e:
        print("Validation error:", e)
    return job
    




if __name__ == '__main__':


    cot_path = 'data/filtered_test_data_cotune-ibm-data-from-sri-slack.csv'

    df = pd.read_csv(cot_path, index_col=None)
    cols_to_use = ['model_name',
                    'method',
                    'number_gpus',
                    'gpu_model',
                    'tokens_per_sample',
                    'batch_size',
                    'is_valid',
                    ]
    # print(df)

    # this is a not valid configuration, I expect that min_gpus==-1
    config_to_test = df[cols_to_use].iloc[2].to_dict() # iloc 2 returns the series so the resulting dict is flat
    config_to_test = df[cols_to_use].iloc[5].to_dict() # this is not valid as well

    
    # this is a valid configuration
    # config_to_test = df[cols_to_use].iloc[1].to_dict() # I expect min_gpus in VALID_N_GPUS

    print(f"Config to test:\n{config_to_test}")
    validated_config = validate_as_jobconfig(config_to_test)



    #predictor = TabularPredictor.load("AutogluonModels/ag-20250821_121652")  # Adjust path if needed
    # assumes you run from root
    predictor = TabularPredictor.load("autoconf/autoconf/AutogluonModels/ag-20250821_121652")  # Adjust path if needed
    valid_n_gpu = VALID_N_GPUS
    min_gpus, m = recommend_min_gpu(validated_config, valid_n_gpu = valid_n_gpu, predictor = predictor)
    print(f"The recommended number_gpus={min_gpus}")



class MinGpuRecommender():
    def __init__(self, predictor, valid_n_gpu :list[int] = VALID_N_GPUS):
        self.predictor = predictor
        self.valid_n_gpu = valid_n_gpu

    def recommend_min_gpu(self, job_config):
        return recommend_min_gpu(job_config, self.predictor, self.valid_n_gpu)
    




# Stale
    # payload_series = df[cols_to_use].iloc[2]
    # payload = df[cols_to_use].iloc[[2]].to_dict()
    # print(type(payload_series))
    # payload_df = df[cols_to_use].iloc[[2]]