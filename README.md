# Instacart Market Basket Analysis Dashboard

An interactive dashboard built with Dash and Plotly to analyze Instacart market basket data. The dashboard provides insights into product popularity and department distribution across orders.

## Features

- Interactive department selection
- Product count filtering
- Top products visualization by department
- Department distribution analysis
- SQLite database backend for efficient data management

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/instacart-market-basket-analysis.git
cd instacart-market-basket-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download the Instacart Market Basket Analysis dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
   - Place the following CSV files in the project root:
     - `products.csv`
     - `departments.csv`
     - `orders.csv`
     - `order_products__train.csv`

5. Create the database:
```bash
python create_db.py
```

6. Run the application:
```bash
python app.py
```

The dashboard will be available at `http://localhost:8050`

## Project Structure

- `app.py`: Main Dash application
- `create_db.py`: Database creation and data import script
- `requirements.txt`: Python dependencies
- `instacart.db`: SQLite database (created after running create_db.py)

## Dependencies

- dash
- plotly
- pandas
- numpy
- sqlite3

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests! 