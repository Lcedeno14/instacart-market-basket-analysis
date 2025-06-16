# Instacart Market Basket Analysis

A comprehensive market basket analysis and business intelligence platform for Instacart data, featuring scalable PySpark processing, advanced analytics, interactive Dash visualizations, REST API, and financial optimization insights.

## 🎯 Overview

This project analyzes Instacart's retail data to discover product associations, optimize revenue, and provide actionable business insights. It features scalable data processing with PySpark, advanced market basket analysis using FP-Growth algorithm, financial optimization with price data, and a modern multi-tab dashboard for business and technical users.

## 🚀 Key Features

### Core Analytics
- **Scalable PySpark Analysis**: Handles large datasets efficiently with optimized processing
- **FP-Growth Algorithm**: Fast frequent pattern mining for market basket analysis
- **Financial Optimization**: Revenue-optimized ranking with price data integration
- **Advanced Metrics**: Support, confidence, lift, and custom business metrics
- **Data Quality Checks**: Automated validation and quality assurance

### Interactive Dashboard
- **Executive Dashboard**: High-level KPIs, weekly/hourly trends, business activity heatmaps
- **Product & Category Performance**: Top products, reordered products, department/aisle analysis
- **Customer Behavior & Reordering**: Reorder rates, customer segmentation, order frequency analysis
- **Market Basket Analysis**: Product association rules, revenue impact, recommendation strategies
- **Price Analysis**: Price distribution, trends, elasticity, and optimization opportunities
- **Data Stories**: Narrative insights and key findings with automated storytelling

### Technical Features
- **REST API**: Programmatic access to data and analytics via Flask Blueprint
- **Database Integration**: PostgreSQL storage with optimized ETL pipeline
- **Deployment Ready**: Heroku/Railway deployment configuration with Gunicorn
- **Comprehensive Logging**: Structured logging with configurable levels
- **Error Handling**: Robust error handling and graceful degradation

## 📊 Dashboard Tabs

### 1. Executive Dashboard
- Real-time KPIs and business metrics
- Weekly and hourly order trends
- Department performance heatmaps
- Customer activity patterns

### 2. Product & Category Performance
- Top performing products and categories
- Reorder rate analysis
- Department and aisle performance
- Product lifecycle insights

### 3. Customer Behavior
- Customer segmentation analysis
- Reorder patterns and frequency
- Customer lifetime value metrics
- Behavioral clustering

### 4. Market Basket Analysis
- Association rule mining results
- Product recommendation engine
- Revenue impact analysis
- Cross-selling opportunities

### 5. Price Analysis
- Price distribution and trends
- Price elasticity analysis
- Revenue optimization insights
- Margin opportunity identification

### 6. Data Stories
- Automated narrative insights
- Key business findings
- Trend explanations
- Actionable recommendations

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11.7+
- PySpark 3.5.0
- PostgreSQL database
- Node.js (for development tools)

### Quick Start

1. **Clone and install dependencies:**
   ```bash
   git clone <repository-url>
   cd instacart-market-basket-analysis
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Create .env file
   DATABASE_URL=postgresql://user:password@localhost:5432/instacart_db
   ```

3. **Initialize database:**
   ```bash
   python setup_database.py
   python create_db.py
   ```

4. **Run ETL pipeline (if needed):**
   ```bash
   python etl_pipeline.py
   ```

5. **Start the dashboard:**
   ```bash
   python app.py
   ```
   The dashboard will be available at http://localhost:8050

### Development Setup

1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Run tests:**
   ```bash
   pytest tests/
   ```

## 📁 Project Structure

```
instacart-market-basket-analysis/
├── src/
│   ├── analysis/
│   │   ├── data_processor.py      # Core data processing logic
│   │   ├── data_storytelling.py   # Automated insights generation
│   │   └── metrics.py             # Business metrics calculations
│   ├── tabs/
│   │   ├── executive_dashboard_tab.py
│   │   ├── product_category_performance_tab.py
│   │   ├── customer_behavior_tab.py
│   │   ├── market_basket_tab.py
│   │   ├── price_analysis_tab.py
│   │   └── data_stories_tab.py
│   ├── utils/
│   │   └── logging_config.py      # Logging configuration
│   ├── assets/                    # Static assets
│   └── visualization/             # Visualization utilities
├── data/                          # Data files and cache
├── logs/                          # Application logs
├── tests/                         # Test suite
├── app.py                         # Main Dash application
├── api.py                         # REST API endpoints
├── main.py                        # Price generation script
├── etl_pipeline.py                # ETL data processing
├── setup_database.py              # Database initialization
├── create_db.py                   # Database schema creation
├── data_quality.py                # Data quality checks
├── financial_analysis.py          # Financial optimization analysis
├── run_full_analysis.py           # Full analysis runner
├── spark_market_basket_simple.py  # PySpark market basket analysis
├── check_rules.py                 # Rule validation utility
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── Procfile                       # Heroku deployment config
├── runtime.txt                    # Python version specification
└── .env                          # Environment variables
```

## 🔌 API Endpoints

The application provides a REST API for programmatic access to data and analytics:

### Core Endpoints
- `GET /api/health` - Health check and database status
- `GET /api/products` - Product information
- `GET /api/products/{product_id}` - Specific product details
- `GET /api/products/department/{department_id}` - Products by department
- `GET /api/orders` - Order data
- `GET /api/orders/{order_id}` - Specific order details
- `GET /api/orders/user/{user_id}` - User's order history

### Analytics Endpoints
- `GET /api/analytics/departments` - Department performance analytics
- `GET /api/analytics/customers` - Customer behavior analytics
- `GET /api/analytics/orders/summary` - Order summary statistics

### Usage Examples

```python
import requests
import pandas as pd

# Fetch top products for a department
response = requests.get('http://localhost:8050/api/products/department/1')
products_df = pd.DataFrame(response.json())

# Get customer analytics
response = requests.get('http://localhost:8050/api/analytics/customers')
customer_insights = response.json()
```

## 🔧 Advanced Features

### Market Basket Analysis
- **FP-Growth Algorithm**: Efficient frequent pattern mining
- **Association Rules**: Support, confidence, and lift calculations
- **Revenue Optimization**: Price-weighted rule ranking
- **Financial Impact**: Revenue potential and AOV impact analysis

### Data Processing
- **ETL Pipeline**: Automated data extraction, transformation, and loading
- **Data Quality**: Automated validation and quality checks
- **Caching**: Optimized data caching for performance
- **Logging**: Comprehensive logging with configurable levels

### Financial Analysis
- **Price Integration**: Product pricing with AI-generated price estimates
- **Revenue Optimization**: Rule-based revenue maximization
- **Margin Analysis**: Profit margin optimization opportunities
- **Statistical Analysis**: Advanced statistical insights

## 🚀 Deployment

### Heroku/Railway Deployment
The application is configured for cloud deployment with:
- `Procfile` for Gunicorn server
- `runtime.txt` for Python version specification
- Environment variable configuration
- Database URL configuration

### Environment Variables
```bash
DATABASE_URL=postgresql://user:password@host:port/database
```

## 📈 Analysis Capabilities

### Full Analysis Runner
```bash
python run_full_analysis.py
```
Runs comprehensive analysis on all orders with detailed logging and comparison.

### Financial Optimization
```bash
python financial_analysis.py
```
Performs detailed financial analysis of market basket rules with price optimization.

### Data Quality Checks
```bash
python data_quality.py
```
Validates data integrity and quality across all datasets.

## 🧪 Development

### Code Quality
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pylint**: Code analysis
- **Pre-commit**: Automated quality checks

### Testing
- **Pytest**: Test framework
- **Coverage**: Test coverage reporting
- **Mock**: Test mocking utilities

### Documentation
- **Sphinx**: Documentation generation
- **MkDocs**: Alternative documentation
- **API Documentation**: Comprehensive API docs

## 📚 Additional Documentation

- [API Documentation](API_DOCUMENTATION.md) - Complete API reference
- [Database Schema](DATABASE_SCHEMA.md) - Database structure and relationships
- [Spark Optimization Journey](SPARK_OPTIMIZATION_JOURNEY.md) - Performance optimization details
- [Market Basket Improvements](MARKET_BASKET_IMPROVEMENTS.md) - Analysis improvements

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code quality standards are met
5. Update documentation as needed
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include docstrings for all public functions
- Write tests for new features
- Update relevant documentation

## 📄 License

MIT License. See LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs in the `logs/` directory
2. Review the documentation files
3. Check the API health endpoint: `/api/health`
4. Open an issue with detailed error information

---

**Built with ❤️ using PySpark, Dash, PostgreSQL, and modern Python technologies.** 