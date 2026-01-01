from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    s3_bucket: str = "mlops-lab11-models-jacek-tysz"

    s3_classifier_key: str = "classifier.joblib"
    s3_sentence_transformer_prefix: str = "sentence_transformer.model"

    artifacts_dir: Path = Path("artifacts")
    classifier_joblib_path: Path = artifacts_dir / "classifier.joblib"
    sentence_transformer_dir: Path = artifacts_dir / "sentence_transformer.model"

    onnx_dir: Path = Path("onnx_models")
    onnx_embedding_model_path: Path = onnx_dir / "sentence_embeddings.onnx"
    onnx_classifier_path: Path = onnx_dir / "classifier.onnx"
    onnx_tokenizer_path: Path = onnx_dir / "tokenizer" / "tokenizer.json"

    embedding_dim: int = 384
