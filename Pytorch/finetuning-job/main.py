from utils.training_utils import TrainingUtils
from oip_tracking_client.tracking import TrackingClient
import os
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

TrackingClient.connect()

experiment_name = "Jobs Test 1"
TrackingClient.set_experiment(experiment_name)


def push_to_huggingface(output_dir: str = "./output"):
    """
    The repository is already created, so we just need to push the files to it.
    Args:
        output_dir (str): Directory containing files to upload (default: "./output")
    """
    print("Pushing to Hugging Face... 🤗")

    token = os.getenv("HF_TOKEN")
    model_id = os.getenv("RESULT_MODEL_ID")

    api = HfApi(token=token)
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist")
    
    if not os.listdir(output_dir):
        raise ValueError(f"Output directory '{output_dir}' is empty")
    
    try:
        upload_folder(
            folder_path=output_dir,
            repo_id=model_id,
            repo_type="model",
            token=token,
            commit_message=f"Upload model files from {output_dir}",
            ignore_patterns=[".git", ".gitignore", "__pycache__", "*.pyc", ".DS_Store"]
        )
        
        print(f"Successfully uploaded files from '{output_dir}' to {model_id}")
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        return None

def train(backbone_model, dataset_name, dataset_task):
    training_utils = TrainingUtils(
        backbone_model=backbone_model,
        dataset_name=dataset_name,
        dataset_task=dataset_task,
    )

    trainer = training_utils.get_trainer(
        per_device_train_batch_size=64, per_device_eval_batch_size=64, num_train_epochs=1
    )

    trainer.train()

    print("=" * 100)
    print("Training complete ✅")
    print("=" * 100)


def main():
    backbone_model = "bert-base-uncased"
    dataset_name = "glue"
    dataset_task = "sst2"

    # make sure that HF repo exists, if not create it and if you get an error, print the error
    repo_url = create_repo(
            repo_id=os.getenv("RESULT_MODEL_ID"),
            repo_type="model",
            private=True,
            token=os.getenv("HF_TOKEN"),
            exist_ok=True 
        )
        
    print(f"Repository created/accessed: {repo_url}")

    # if repo_url is None, raise an error
    if repo_url is None:
        raise ValueError("Repository not created")

    with TrackingClient.start_run():
        run_name = os.getenv("RUN_NAME")
        TrackingClient.set_run_name(run_name)

        train(backbone_model, dataset_name, dataset_task)

    push_to_huggingface()
    TrackingClient.log_param("repo_url", repo_url)
    print("Model pushed to Hugging Face 🤗")


if __name__ == "__main__":
    main()
