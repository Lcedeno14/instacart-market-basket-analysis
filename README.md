# Instacart Market Basket Analysis Dashboard

A comprehensive data analysis and visualization dashboard for Instacart order data, featuring market basket analysis, customer segmentation, data storytelling, and interactive visualizations.

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
- Interactive rule exploration with lift, support, and confidence metrics

### Customer Segmentation
- RFM (Recency, Frequency, Monetary) analysis
- Purchase pattern clustering
- Department preference analysis
- Interactive cluster visualization
- Customizable number of segments

### Data Storytelling
- Customer journey analysis with shopping patterns
- Seasonal trends and time-based insights
- Price analysis and customer spending behavior
- Product association narratives
- Interactive visualizations for each story
- Actionable business recommendations

## Technical Stack
- **Frontend**: Dash/Plotly for interactive visualizations
- **Backend**: Python with SQLAlchemy for database operations
- **Data Analysis**: Pandas, NumPy, scikit-learn
- **Market Basket Analysis**: mlxtend
- **Database**: PostgreSQL (production) / SQLite (development)
- **Testing**: pytest for unit testing
- **Development**: Black for code formatting, flake8 for linting

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Lcedeno14/instacart-market-basket-analysis.git
cd instacart-market-basket-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

4. Set up the database:
   - For development, the app will use SQLite by default (instacart.db)
   - For production, set the DATABASE_URL environment variable:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/instacart"
```

5. Run the application:
```bash
python app.py
```

The dashboard will be available at http://localhost:8050

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Format code:
```bash
black .
```

4. Run linter:
```bash
flake8 .
```

## Data Engineering Features
- Efficient data loading and caching
- Optimized SQL queries for large datasets
- Data quality checks and validation
- Scalable database design
- Production-ready deployment configuration
- Comprehensive logging and error handling
- Database migration support

## Data Analysis Features
- Market basket analysis with multiple algorithms
- Customer segmentation using RFM and clustering
- Time-based pattern analysis
- Department-level insights
- Price-weighted analysis
- Data storytelling with narrative insights
- Interactive visualizations for all analyses
- Automated business recommendations

## Project Structure
```
instacart-market-basket-analysis/
├── src/
│   └── analysis/
│       └── data_storytelling.py    # Data storytelling module
├── tests/
│   └── test_customer_segmentation.py  # Unit tests
├── app.py                          # Main dashboard application
├── api.py                          # REST API endpoints
├── customer_segmentation.py        # Customer segmentation logic
├── data_quality.py                 # Data quality checks
├── etl_pipeline.py                 # ETL operations
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
└── README.md                       # Project documentation
```

## Future Enhancements
- Real-time data updates
- Automated model retraining
- Additional segmentation methods
- Export functionality for analysis results
- API endpoints for data access
- Additional data storytelling narratives
- Automated report generation
- Integration with BI tools

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:
1. Add tests for new features
2. Update documentation
3. Follow the code style (Black formatting)
4. Run the test suite before submitting

## License
This project is licensed under the MIT License - see the LICENSE file for details. 