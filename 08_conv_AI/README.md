This directory contains files and folders related to the conversational AI component of the project, including:

- **.streamlit:** Configuration files for Streamlit, used for deploying the conversational AI interface.
- **chatbot:** Scripts and resources for building and managing the chatbot functionality.
- **img:** Images and graphical assets used in the conversational AI interface.
- **predictions:** Files related to generating and managing predictions made by the AI model.
- **sentiment:** Scripts and data for sentiment analysis, which the AI uses to interpret user input.
- **service:** Service layer scripts that handle interactions between the AI model and the user interface.
- **test_data:** Test data used to validate the functionality of the conversational AI.
- **Dockerfile:** Configuration file for containerizing the application using Docker.
- **README.md:** Documentation file that provides an overview and instructions for the conversational AI component.
- **index.py:** Main script for running the conversational AI application.
- **requirements.txt:** List of Python dependencies required to run the application.
- **style.css:** Stylesheet used to customize the appearance of the Streamlit interface.

These components together enable the deployment and operation of the conversational AI, providing users with a seamless interface for interacting with the AI model.


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
