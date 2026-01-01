from pathlib import Path

import boto3

from settings import Settings


def download_artifacts() -> None:
    settings = Settings()

    s3 = boto3.client("s3", region_name=settings.aws_region)

    Path(settings.classifier_joblib_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.sentence_transformer_dir).mkdir(parents=True, exist_ok=True)

    s3.download_file(
        settings.s3_bucket,
        settings.s3_classifier_key,
        str(settings.classifier_joblib_path),
    )

    prefix = settings.s3_sentence_transformer_prefix.rstrip("/") + "/"

    response = s3.list_objects_v2(
        Bucket=settings.s3_bucket,
        Prefix=prefix,
    )

    for obj in response.get("Contents", []):
        key = obj["Key"]
        relative = key[len(prefix):]
        if not relative:
            continue

        local_path = Path(settings.sentence_transformer_dir) / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3.download_file(settings.s3_bucket, key, str(local_path))

    print("All artifacts downloaded.")


if __name__ == "__main__":
    download_artifacts()
