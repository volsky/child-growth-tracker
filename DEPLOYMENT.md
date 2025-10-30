# Google App Engine Deployment Guide

This guide will help you deploy the Child Growth Tracker to Google App Engine.

## Prerequisites

1. **Google Cloud Account**: Create a Google Cloud account at https://cloud.google.com
2. **Google Cloud SDK**: Install the gcloud CLI tool
   ```bash
   # macOS
   brew install --cask google-cloud-sdk

   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

3. **Enable Billing**: Make sure billing is enabled for your Google Cloud project

## Setup Steps

### 1. Initialize Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (or use existing one)
gcloud projects create child-growth-tracker-PROJECT_ID
# Replace PROJECT_ID with a unique identifier

# Set the project
gcloud config set project child-growth-tracker-PROJECT_ID

# Enable required APIs
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Create App Engine Application

```bash
# Initialize App Engine (choose your preferred region)
gcloud app create --region=us-central
# Other regions: europe-west, asia-northeast1, etc.
```

### 3. Deploy the Application

```bash
# From the project directory, deploy the app
gcloud app deploy

# When prompted, confirm the deployment
```

The deployment process will:
- Upload your code to Google Cloud
- Install dependencies from requirements.txt
- Start the Streamlit application
- Make it available at: https://child-growth-tracker-PROJECT_ID.appspot.com

### 4. View Your Application

```bash
# Open the deployed app in your browser
gcloud app browse
```

### 5. View Logs

```bash
# Stream logs in real-time
gcloud app logs tail -s default

# View logs in Cloud Console
gcloud app logs read
```

## Cost Optimization

The `app.yaml` configuration includes:
- **Automatic Scaling**: Scales from 1 to 3 instances based on traffic
- **Resource Limits**: 1 CPU, 2GB RAM per instance
- **Flexible Environment**: Best for Streamlit apps

### Estimated Costs
- **Low traffic** (< 1000 visits/month): ~$50-75/month
- **Medium traffic** (1000-10000 visits/month): ~$100-200/month

To reduce costs:
1. Use manual scaling instead of automatic:
   ```yaml
   manual_scaling:
     instances: 1
   ```

2. Stop the app when not in use:
   ```bash
   gcloud app versions stop VERSION_ID
   ```

3. Use standard environment (requires different configuration)

## Update Deployment

To deploy updates:

```bash
# Pull latest changes
git pull origin main

# Deploy
gcloud app deploy
```

## Troubleshooting

### Deployment Fails
```bash
# Check detailed logs
gcloud app logs tail -s default

# Validate app.yaml
gcloud app deploy --dry-run
```

### App Not Responding
```bash
# Check health
gcloud app browse

# Restart by deploying again
gcloud app deploy
```

### Out of Memory
Increase memory in `app.yaml`:
```yaml
resources:
  memory_gb: 4  # Increase from 2 to 4
```

## Custom Domain

To use a custom domain:

```bash
# Add custom domain
gcloud app domain-mappings create www.yourdomain.com

# Follow the instructions to update DNS records
```

## Continuous Deployment

To set up automatic deployment from GitHub:

1. Go to Google Cloud Console > Cloud Build > Triggers
2. Click "Create Trigger"
3. Connect your GitHub repository
4. Set trigger to run on push to main branch
5. Use the cloudbuild.yaml configuration (create if needed)

## Security

The app is configured with:
- XSRF protection enabled
- CORS disabled for security
- Headless mode for production
- Health checks for monitoring

## Monitoring

View metrics in Google Cloud Console:
- App Engine > Dashboard
- Monitor: CPU usage, memory, requests, latency
- Set up alerts for high CPU/memory usage

## Support

For issues:
1. Check logs: `gcloud app logs tail`
2. Review App Engine documentation: https://cloud.google.com/appengine/docs
3. Check Streamlit documentation: https://docs.streamlit.io

## Useful Commands

```bash
# List versions
gcloud app versions list

# Stop a version
gcloud app versions stop VERSION_ID

# Delete a version
gcloud app versions delete VERSION_ID

# Set traffic split
gcloud app services set-traffic default --splits VERSION_ID=1.0

# View current project
gcloud config get-value project

# SSH into instance (for debugging)
gcloud app instances ssh INSTANCE_ID --service=default
```
