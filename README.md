# Instacart Market Basket Analysis Dashboard

A comprehensive data analysis and visualization dashboard for Instacart order data, featuring market basket analysis, customer segmentation, and interactive visualizations.

## Features

### Basic Analysis
- Interactive filtering by department and day of week
- Top products visualization
- Department distribution analysis
- Order frequency heatmap by day and hour

### Market Basket Analysis
- Product association rules using Apriori algorithm
- Department-level associations using FP-Growth
- Price-weighted analysis for high-value product combinations
- Adjustable support and confidence thresholds

### Customer Segmentation
- RFM (Recency, Frequency, Monetary) analysis
- Purchase pattern clustering
- Department preference analysis
- Interactive cluster visualization

## Technical Stack
- **Frontend**: Dash/Plotly for interactive visualizations
- **Backend**: Python with SQLAlchemy for database operations
- **Data Analysis**: Pandas, NumPy, scikit-learn
- **Market Basket Analysis**: mlxtend
- **Database**: PostgreSQL (production) / SQLite (development)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/instacart-market-basket-analysis.git
cd instacart-market-basket-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
   - For development, the app will use SQLite by default
   - For production, set the DATABASE_URL environment variable:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/instacart"
```

5. Run the application:
```bash
python app.py
```

The dashboard will be available at http://localhost:8050

## Data Engineering Features
- Efficient data loading and caching
- Optimized SQL queries for large datasets
- Data quality checks and validation
- Scalable database design
- Production-ready deployment configuration

## Data Analysis Features
- Market basket analysis with multiple algorithms
- Customer segmentation using RFM and clustering
- Time-based pattern analysis
- Department-level insights
- Price-weighted analysis

## Future Enhancements
- Real-time data updates
- Automated model retraining
- Additional segmentation methods
- Export functionality for analysis results
- API endpoints for data access

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 