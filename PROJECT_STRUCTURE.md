# Project Structure

This document outlines the organization of the Stock Analysis App project directory.

## Core Application Components
- **`streamlit_app.py`**: The main entry point for the Streamlit dashboard.
- **`modules/`**: Contains the core logic of the application.
  - **`modules/data/`**: Data fetching and handling logic (e.g., historical data, fundamentals).
    - **`scrapers/`**: All web scraping scripts for fetching news, comments, and other unstructured data.
  - **`modules/models/`**: Machine learning models (e.g., Sentiment analysis, LSTM, XGBoost).
  - **`modules/ui/`**: User interface components and page layouts for Streamlit.
  - **`modules/utils/`**: Helper functions and configuration loaders.
- **`pages/`**: Streamlit multi-page routing files.

## Environment & Configuration
- **`.env`** and **`.env.example`**: Environment variable definitions.
- **`config.py`**: Global configuration settings.
- **`requirements.txt`**: Python package dependencies.

## Data Storage
- **`data/`**: Local storage for all structured data (CSVs, JSONs). Note: This directory is typically gitignored.
  - **`news_history/`**: Stores scraped news articles per ticker.
- **`models/`**: Saved trained model weights and preprocessors.
- **`portfolios/`**: User portfolio data.
- **`logs/`**: Application runtime logs.

## Background Scripts & Automation (Orchestration)
- **`scripts/`**: Reorganized directory containing all background tasks:
  - **`scripts/data_collection/`**: Scripts responsible for mass data acquisition (RSS scraping, complete content fetching, YouTube scraping).
  - **`scripts/reporting/`**: Scripts that generate the PDF reports and documents.
  - **`scripts/training/`**: Scripts for bulk training models.

## Documentation & Assets
- **`docs/`**: Detailed technical documentation and recreation guides.
- **`report/`** & **`reports/`**: Output directories for generated PDFs and analysis reports.
- **`figures_gen/`**: Generated chart image assets for reports.

## Legacy / Transition Folders
- **`Universal_Engine_Workspace/`**: Former workspace for testing massive data processing and report generation. Contents are being migrated to `scripts/` and `modules/`.
