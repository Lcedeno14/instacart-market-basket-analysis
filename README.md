# Instacart Market Basket Analysis

A comprehensive market basket analysis and business intelligence dashboard for Instacart data, using PySpark, FP-Growth, and interactive Dash visualizations.

## Overview

This project analyzes Instacart's retail data to discover product associations, optimize revenue, and provide actionable business insights. It features scalable data processing, advanced analytics, and a modern dashboard for business and technical users.

## Key Features
- **Scalable PySpark Analysis**: Handles large datasets efficiently
- **FP-Growth Algorithm**: Fast frequent pattern mining
- **Revenue-Optimized Ranking**: Post-processing with price data
- **Interactive Dash Dashboard**: Visualizes trends, product performance, and customer behavior
- **Database Integration**: PostgreSQL storage and ETL pipeline

## Main Dashboards

- **Executive Dashboard**: High-level KPIs, weekly/hourly trends, business activity heatmaps
- **Product & Category Performance**: Top products, reordered products, department/aisle analysis
- **Customer Behavior & Reordering**: Reorder rates, customer segmentation, order frequency
- **Market Basket Analysis**: Product association rules, revenue impact, recommendation strategies
- **Price Analysis**: Price distribution, trends, and elasticity
- **Data Stories**: Narrative insights and key findings

## Usage

### Prerequisites
- Python 3.11+
- PySpark 3.x
- PostgreSQL
- Required packages: `pyspark`, `sqlalchemy`, `pandas`, `tqdm`, `dash`, `plotly`

### Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up the database (recommended):
   ```bash
   python setup_database.py
   ```
3. Start the dashboard:
   ```bash
   python app.py
   ```
   The dashboard will be available at http://localhost:8050

## Project Structure
```
instacart-market-basket-analysis/
├── src/
│   ├── analysis/
│   │   ├── data_processor.py
│   │   ├── data_storytelling.py
│   │   └── metrics.py
│   ├── tabs/
│   │   ├── executive_dashboard_tab.py
│   │   ├── product_category_performance_tab.py
│   │   ├── customer_behavior_tab.py
│   │   ├── market_basket_tab.py
│   │   ├── price_analysis_tab.py
│   │   └── data_stories_tab.py
│   └── utils/
│       └── logging_config.py
├── data/
├── logs/
├── tests/
├── app.py
├── api.py
├── etl_pipeline.py
├── setup_database.py
├── create_db.py
├── data_quality.py
└── requirements.txt
```

## API Endpoints
- `/api/orders` - Order data
- `/api/products` - Product information
- `/api/departments` - Department data
- `/api/health` - Health check

## Development
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Run tests: `pytest tests/`
- Format code: `black .`
- Lint: `flake8 .`

## Contributing
Contributions are welcome! Please add tests, update documentation, and follow code style guidelines.

## License
MIT License. See LICENSE file for details. 