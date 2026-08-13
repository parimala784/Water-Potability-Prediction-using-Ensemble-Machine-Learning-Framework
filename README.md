# Water Quality Prediction & Visualization

An end-to-end machine learning project for predicting water potability using physicochemical parameters. Features a modern web application with multiple ML models, interactive visualizations, and a dataset analytics dashboard.

## Features

- **Prediction**: Enter 9 water quality parameters → get potability prediction with probability score
- **Multiple Models**: Random Forest, Balanced RF, XGBoost, CatBoost, LightGBM, SVM, Extra Trees, MLP, Ensemble, Stacking
- **Best Model Default**: Automatically selects the best-performing model after training
- **Dataset Dashboard**: View statistics, feature averages, and water quality guidelines
- **Graphs**: Correlation heatmaps, distributions, 2D/3D scatter plots
- **Chatbot**: 100+ Q&A about water quality and the project
- **Dark/Light Theme**: Toggle with persisted preference

## Quick Start

### Run both frontend and backend at once

```bash
# First time: install dependencies
cd backend
python -m venv venv
venv\Scripts\activate   # Windows; on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py
cd ..
cd frontend && npm install && cd ..

# Run both
python run.py
```

- Backend: http://localhost:8000  
- Frontend: http://localhost:5173  
- Press **Ctrl+C** to stop both.

### Run separately

**Backend:**
```bash
cd backend
python -m venv venv     # if venv doesn't exist
venv\Scripts\activate   # Windows; on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt   # if not already installed
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 (Vite). Ensure the backend is running on http://localhost:8000.

### Notebooks

```bash
cd backend
jupyter notebook notebooks/
```

Run in order: 01_eda → 02_preprocessing → 03_visualization → 04_model_building → 05_model_comparison.

## Project Structure

```
water quality forcasting and visualization/
├── images/                  # Confusion matrices, model comparison (run save_model_figures.py)
├── backend/                 # Python API & ML
│   ├── app/                 # FastAPI app, config, predict, features, stacking
│   ├── data/                # water_potability.csv
│   ├── models/              # Trained .pkl files, scaler, best_model.json
│   ├── notebooks/           # EDA, preprocessing, visualization, model building, comparison
│   ├── notebook_outputs/    # Figures from notebooks
│   ├── train.py             # Training script (SMOTE, jitter, all models)
│   └── requirements.txt
├── frontend/                # React + TypeScript (Vite)
│   └── src/
│       ├── components/      # PredictionForm, PredictionResult, Charts, Chatbot, etc.
│       ├── pages/           # Home, Prediction, Graphs, Dashboard
│       ├── api/             # API client
│       └── context/         # Theme
└── doc/                     # Project documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict potability. Body: JSON with 9 parameters. Query: `model_id` (optional) |
| `/api/models` | GET | List models + default_model_id |
| `/api/stats` | GET | Dataset statistics, feature means/stds |
| `/api/features` | GET | Feature names |

## Water Quality Parameters

| Parameter | Description |
|-----------|-------------|
| ph | Acidity/alkalinity (6.5–8.5 optimal) |
| Hardness | Mineral content (mg/L) |
| Solids | Total dissolved solids (mg/L) |
| Chloramines | Disinfectant level (mg/L) |
| Sulfate | Sulfate concentration (mg/L) |
| Conductivity | Electrical conductivity (µS/cm) |
| Organic_carbon | Organic matter (ppm) |
| Trihalomethanes | THMs (µg/L) |
| Turbidity | Water clarity (NTU) |

## Documentation

See the **[doc/](doc/)** folder for complete project documentation:

- [Table of Contents](doc/README.md)
- [Abstract](doc/01_abstract.md)
- [Introduction](doc/02_introduction.md)
- [Related Work](doc/03_related_work.md)
- [Methodology](doc/04_methodology.md)
- [Folder Structure](doc/05_folder_structure.md)
- [Figures & Tables](doc/06_figures.md)
- [Conclusion](doc/07_conclusion.md)
- [Future Work](doc/08_future_work.md)
- [References](doc/09_references.md)
- [Images Gallery](doc/10_images_gallery.md)

## Tech Stack

- **Backend**: FastAPI, scikit-learn, XGBoost, CatBoost, LightGBM, imbalanced-learn
- **Frontend**: React, TypeScript, Vite, Recharts, Plotly
- **Notebooks**: Jupyter, matplotlib, seaborn, pandas

## License

MIT

Note: Trained machine learning model files are excluded from this repository because of GitHub's file-size limitations. The repository contains the complete source code, preprocessing pipeline, frontend, backend, documentation, and project resources required to understand and reproduce the system.