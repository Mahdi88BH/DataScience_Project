from src.Wine_Quality_Prediction import logger
from src.Wine_Quality_Prediction.pipeline.data_Ingestion_pipeline import DataIngestionTrainingPipeline
from src.Wine_Quality_Prediction.pipeline.data_Validation_pipeline import DataValidationTrainingPipeline

logger.info("Welcome to our project Data Science Prediction to the Wine Quality")

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e