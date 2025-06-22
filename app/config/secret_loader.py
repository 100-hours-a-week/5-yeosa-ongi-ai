import os
from google.cloud import secretmanager

def load_secrets_from_gcp():
    project_id = os.getenv("PROJECT_ID")
    app_env = os.getenv("APP_ENV")

    if not project_id:
        raise RuntimeError("PROJECT_ID 환경변수가 설정되지 않았습니다.")
    if not app_env:
        raise RuntimeError("APP_ENV 환경변수가 설정되지 않았습니다.")
        
    client = secretmanager.SecretManagerServiceClient()

    secret_map = {
        "AWS_ACCESS_KEY": "fastapi_aws_access_key",
        "AWS_REGION": "fastapi_aws_region",
        "AWS_SECRET_KEY": "fastapi_aws_secret_key",
        "GCP_KEY": "fastapi_gcp_key",
        "GCS_BUCKET_NAME": "fastapi_gcs_bucket_name",
        "IMAGE_MODE": "fastapi_image_mode",
        "LOCAL_IMG_PATH": "fastapi_local_img_path",
        "MODEL_NAME": "fastapi_model_name",
        "REDIS_CACHE_TTL": "fastapi_redis_cache_ttl",
        "REDIS_DB": "fastapi_redis_db",
        "REDIS_PORT": "fastapi_redis_port",
        "S3_BUCKET_NAME": f"fastapi_s3_bucket_name_{app_env}",
        "GPU_SERVER_BASE_URL": f"fastapi_gpu_server_base_url_{app_env}",
        "REDIS_HOST": f"fastapi_redis_host_{app_env}"
    }


    for env_key, secret_id in secret_map.items():
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        os.environ[env_key] = response.payload.data.decode("UTF-8")
