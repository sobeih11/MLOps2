from src.split_data import split_and_save
from src.train import run_tuning
from src.evaluation import evaluate_pipeline
import hydra
from omegaconf import DictConfig
from src.logger import get_logger

logger = get_logger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("🚀 Starting full Titanic ML pipeline...")

    # 1. Split data
    logger.info(" Step 1: Splitting data")
    split_and_save(cfg)

    # 2. Train and tune model
    logger.info("Step 2: Training & Hyperparameter Tuning")
    run_tuning(cfg.pipeline)

    # 3. Evaluate model
    logger.info("Step 3: Evaluating final model")
    evaluate_pipeline(cfg)

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
