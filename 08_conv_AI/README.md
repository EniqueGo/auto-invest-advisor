# Bitcoin Prediction Website


## Before Running

Please copy the file `secrets.toml` from `secret_keys` on Google Drive  to the following path:
```bash
.streamlit/secrets.toml
```

## Generate requirements.txt
```bash
 pigar generate
```

## Run Streamlit
```bash
streamlit run index.py
```

## Cloud Run Deployment

```bash
gcloud config set project adsp-capstone-enique

## Clean-up (to get fresh experience) and remove all unused and dangling images
docker system prune -a -f

# Enable Artifact Registry
gcloud services enable artifactregistry.googleapis.com

# Verify / list repositories
gcloud artifacts repositories list

# Delete repository
gcloud artifacts repositories delete "stonkgo-web" --location=us

gcloud artifacts repositories create "stonkgo-web" --repository-format=docker --location=us --description="stonkgo-web v4"

# Build Docker image
#docker image build -t us-docker.pkg.dev/PROJECT-ID/REPOSITORY/APP-NAME:latest .
docker image build -t us-docker.pkg.dev/adsp-capstone-enique/stonkgo-web/stonkgo-web-app:latest .

# Push Docker image into artifact registry
#docker push us-docker.pkg.dev/PROJECT-ID/REPOSITORY/APP-NAME:latest
docker push us-docker.pkg.dev/adsp-capstone-enique/stonkgo-web/stonkgo-web-app:latest

# If you get an error
gcloud auth configure-docker "us-docker.pkg.dev"

# Enable  Cloud Run API
gcloud services enable run.googleapis.com


gcloud run deploy stonkgo-web-app \
 --image=us-docker.pkg.dev/adsp-capstone-enique/stonkgo-web/stonkgo-web-app:latest \
 --platform managed \
 --allow-unauthenticated \
 --region=us-central1 \
 --project=adsp-capstone-enique
 
# Since we have 403 forbidden error, we need to add the Cloud Run Invoker role to the Cloud Run service account
gcloud run services proxy stonkgo-web-app --project adsp-capstone-enique

```
