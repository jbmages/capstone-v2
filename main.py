import os
import logging
from scripts.workflow import FullWorkflow

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "workflow.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Workflow started.")

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

if __name__ == "__main__":
    try:
        workflow = FullWorkflow(GOOGLE_DRIVE_URL)
        workflow.data_setup()
        workflow.clustering()
        logging.info("Workflow completed successfully.")
    except Exception as e:
        logging.error(f"Workflow failed: {str(e)}", exc_info=True)
