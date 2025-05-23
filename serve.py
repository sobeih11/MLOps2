import litserve as ls
from omegaconf import OmegaConf
from src.deployment.inference_api import InferenceAPI

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    api = InferenceAPI(cfg)
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000, generate_client_file=False)