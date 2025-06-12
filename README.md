# Instacart Market Basket Analysis Dashboard

A comprehensive data analysis and visualization dashboard for Instacart order data, featuring market basket analysis, customer segmentation, data storytelling, and interactive visualizations.

## Features

### Executive Dashboard
- **High-level KPIs**: Total Revenue, Average Order Value, Total Orders, Unique Customers
- **Weekly Trends**: Orders and revenue trends by day of week
- **Hourly Analysis**: Orders and revenue patterns by hour and day of week
- **Business Activity Heatmaps**: Revenue and orders heatmaps (Hour × Day of Week)
- **Executive Insights**: Clear visualization of business performance patterns

### Product & Category Performance Dashboard
- **Top Products Analysis**: Top 10 products by revenue vs. order volume
- **Reordered Products**: Products with highest reorder rates
- **Department Performance**: Revenue and order distribution across departments
- **Aisle Analysis**: Top performing aisles and product categories
- **Price Analysis**: Most expensive products and price distribution

### Customer Behavior & Reordering Dashboard
- **Reorder Rate by Price Quartile**: Do customers reorder expensive items more or less frequently?
- **Cumulative Revenue per Customer (Top 20)**: Revenue concentration among top customers
- **Days Since Prior Order Distribution**: Customer ordering frequency patterns
- **AOV by Hour & Day of Week**: When do customers spend the most?
- **Reorder Count vs. Order Count**: Customer loyalty and reordering behavior analysis

### Market Basket Analysis
- **Association Rules**: Discover product relationships and recommendations
- **Support, Confidence, and Lift**: Key metrics for product associations
- **Interactive Filtering**: Filter by department, minimum support, and confidence
- **Network Visualization**: Visual representation of product relationships

### Basic Analysis
- **Order Patterns**: Orders by day of week and hour of day
- **Department Analysis**: Order distribution across departments
- **Interactive Filters**: Filter by department and time period
- **Trend Analysis**: Temporal patterns in ordering behavior

### Price Analysis
- **Price Distribution**: Histograms and box plots of product prices
- **Price by Department**: Average prices across different departments
- **Price Trends**: Price patterns over time and by category
- **Statistical Analysis**: Price statistics and outliers

### Data Stories
- **Narrative Insights**: Story-driven data analysis
- **Key Findings**: Automated discovery of important patterns
- **Business Implications**: Actionable insights from the data
- **Trend Analysis**: Temporal and categorical patterns

### KPI Dashboard
- **Business KPIs**: Key performance indicators
- **Revenue Metrics**: Total revenue, average order value
- **Customer Metrics**: Customer count, order frequency
- **Product Metrics**: Product performance and reorder rates

## Tab Order

1. Executive Dashboard - High-level business overview
2. Product & Category Performance - Product and category insights
3. Customer Behavior & Reordering - Behavioral patterns & customer segmentation
4. Basic Analysis - Fundamental order and department analysis
5. Market Basket Analysis - Product association and recommendation analysis
6. Price Analysis - Price distribution and analysis
7. Data Stories - Narrative insights and automated analysis
8. Business KPIs - Key performance indicators dashboard

## 📡 API Integration

The application includes a RESTful API for external integrations:

### **Key Benefits:**
- **Mobile Apps**: Native mobile applications can fetch data
- **BI Tools**: Connect Tableau, Power BI, or other analytics platforms  
- **Automation**: Scripts and automated reporting systems
- **External Dashboards**: Embed analytics in third-party applications

### **Quick API Reference:**
```bash
# Products
GET /api/products                    # All products
GET /api/products/{id}              # Specific product
GET /api/products/department/{id}   # Products by department

# Orders  
GET /api/orders                     # All orders
GET /api/orders/{id}                # Specific order
GET /api/orders/user/{id}           # User's orders

# Analytics
GET /api/analytics/departments      # Department analytics
GET /api/analytics/customers        # Customer insights
GET /api/analytics/orders/summary   # Order summary

# Health Check
GET /api/health                     # API status
```

### **Example Usage:**
```python
import requests

# Fetch top products
response = requests.get('http://localhost:8050/api/products')
products = response.json()

# Get customer analytics
response = requests.get('http://localhost:8050/api/analytics/customers')
analytics = response.json()
```

📖 **Full API Documentation**: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed usage examples, benefits, and integration guides.

## Technical Stack
- **Frontend**: Dash/Plotly for interactive visualizations
- **Backend**: Python with SQLAlchemy for database operations
- **Data Analysis**: Pandas, NumPy, scikit-learn
- **Market Basket Analysis**: mlxtend
- **Database**: PostgreSQL (production) / SQLite (development)
- **Testing**: pytest for unit testing
- **Development**: Black for code formatting, flake8 for linting
- **Logging**: Centralized logging with configurable levels
- **ETL Pipeline**: Industry-standard data loading with validation and tracking

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

5. **Set up the database using the ETL pipeline (Recommended):**
```bash
# Use the new ETL pipeline (industry standard)
python setup_database.py
```

   **OR use the legacy setup (not recommended):**
```bash
# Legacy setup (basic, no validation)
python create_db.py
```

6. Run the application:
```bash
python app.py
```

The dashboard will be available at http://localhost:8050

## 🏭 ETL Pipeline vs Legacy Setup

### **Why Use the ETL Pipeline? (Recommended)**

The ETL pipeline (`setup_database.py`) provides industry-standard data loading with:

- ✅ **Smart File Tracking**: Remembers processed files to avoid reprocessing
- ✅ **Data Validation**: Validates schema, types, and constraints before loading
- ✅ **Chunked Loading**: Handles large files efficiently (10,000 rows at a time)
- ✅ **Error Handling**: Detailed error logging and metadata tracking
- ✅ **Referential Integrity**: Maintains proper foreign key relationships
- ✅ **Performance Indexes**: Automatically creates database indexes
- ✅ **Change Detection**: Uses MD5 hashes to detect file changes

### **Legacy Setup (`create_db.py`)**

The legacy setup is simpler but lacks:
- ❌ No file tracking (always reprocesses)
- ❌ Basic validation only
- ❌ Loads entire files at once (can crash with large files)
- ❌ Limited error handling
- ❌ No metadata tracking

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
- **Industry-Standard ETL Pipeline**: Robust data loading with validation and tracking
- **Centralized Data Processing**: `DataProcessor` class for efficient data loading and merging
- **Optimized Data Loading**: Pre-merged data to avoid redundant operations
- **Data Quality Checks**: Comprehensive validation and error handling
- **Caching System**: Performance optimization with intelligent caching
- **Scalable Architecture**: Modular design for easy maintenance and scaling
- **Production-Ready Logging**: Centralized logging with configurable levels
- **Database Migration Support**: Flexible database configuration

## Data Analysis Features
- **Market Basket Analysis**: Multiple algorithms (Apriori, FP-Growth) with price weighting
- **Time-Based Pattern Analysis**: Seasonal trends and temporal insights
- **Department & Aisle Analysis**: Hierarchical category performance insights
- **Price Analysis**: Elasticity, distribution, and optimization insights
- **Data Storytelling**: Narrative insights with actionable recommendations
- **Interactive Visualizations**: Real-time charts and dashboards
- **Automated Business Recommendations**: Data-driven strategic insights

## Project Structure
```
instacart-market-basket-analysis/
├── src/
│   ├── analysis/
│   │   ├── data_processor.py       # Central data processing and loading
│   │   ├── data_storytelling.py    # Narrative insights and storytelling
│   │   └── metrics.py              # Business metrics and KPIs
│   ├── tabs/
│   │   ├── executive_dashboard_tab.py           # Executive-level insights
│   │   ├── product_category_performance_tab.py  # Product & category analysis
│   │   ├── kpi_dashboard_tab.py                 # KPI tracking
│   │   ├── basic_analysis_tab.py                # Basic data exploration
│   │   ├── market_basket_tab.py                 # Association analysis
│   │   ├── price_analysis_tab.py                # Price analysis and insights
│   │   └── data_stories_tab.py                  # Data storytelling
│   ├── visualization/
│   │   └── kpi_dashboard.py       # KPI dashboard implementation
│   └── utils/
│       └── logging_config.py      # Centralized logging configuration
├── data/                          # CSV data files
├── logs/                          # Application logs
├── tests/                         # Unit tests
├── app.py                         # Main Dash application
├── api.py                         # RESTful API endpoints
├── etl_pipeline.py                # Industry-standard ETL pipeline
├── setup_database.py              # Database setup using ETL pipeline
├── create_db.py                   # Legacy database setup (not recommended)
├── data_quality.py                # Data quality validation
└── requirements.txt               # Python dependencies
```

## ETL Pipeline Features

### **File Processing**
- **Smart Detection**: Only processes files that have changed
- **Batch Loading**: Handles large files in manageable chunks
- **Error Recovery**: Continues processing even if one file fails

### **Data Validation**
- **Schema Validation**: Ensures data matches expected structure
- **Type Checking**: Validates data types match database schema
- **Constraint Validation**: Checks for null values in primary keys
- **Referential Integrity**: Maintains foreign key relationships

### **Metadata Tracking**
- **Processing History**: Tracks when each file was processed
- **File Integrity**: Uses MD5 hashes to detect changes
- **Error Logging**: Detailed error messages and status tracking
- **Performance Metrics**: Row counts and processing times

### **Production Features**
- **PostgreSQL Compatible**: Full support for production databases
- **Index Optimization**: Automatic creation of performance indexes
- **Transaction Safety**: Ensures data consistency
- **Scalable Architecture**: Handles growing data volumes

## Troubleshooting

### **Common Issues**

1. **Database Connection Errors**
   - Ensure `DATABASE_URL` is set correctly
   - Check database server is running
   - Verify network connectivity

2. **Missing CSV Files**
   - Ensure all required files are in the `data/` directory
   - Check file names match expected format
   - Verify file permissions

3. **ETL Pipeline Errors**
   - Check logs in `logs/etl.log` for detailed error messages
   - Verify CSV file format and content
   - Ensure database has proper permissions

4. **Performance Issues**
   - Use the ETL pipeline for better performance
   - Check database indexes are created
   - Monitor memory usage with large files

### **Getting Help**

- Check the logs in the `logs/` directory
- Review error messages in the terminal output
- Ensure all dependencies are installed
- Verify database connection and permissions

## Key Improvements
- **Modular Architecture**: Separated concerns with dedicated tab modules
- **Performance Optimization**: Pre-calculated metrics and intelligent caching
- **Real Data Integration**: Actual price data instead of proxy calculations
- **Comprehensive Logging**: Centralized logging with proper error handling
- **API Integration**: RESTful endpoints for external integrations
- **Executive Dashboards**: High-level business insights and KPIs
- **Product Performance Analysis**: Revenue vs. volume optimization insights

## Future Enhancements
- Real-time data updates and streaming
- Automated model retraining and A/B testing
- Additional segmentation methods and algorithms
- Export functionality for analysis results
- Advanced API endpoints for data access
- Additional data storytelling narratives
- Automated report generation and scheduling
- Integration with BI tools and data warehouses
- Machine learning model deployment
- Real-time alerting and notifications

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:
1. Add tests for new features
2. Update documentation
3. Follow the code style (Black formatting)
4. Run the test suite before submitting
5. Update logging configuration if needed

## License
This project is licensed under the MIT License - see the LICENSE file for details. 