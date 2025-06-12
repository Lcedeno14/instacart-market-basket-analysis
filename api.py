from flask import Blueprint, jsonify, request
from sqlalchemy import create_engine, text
import pandas as pd
import os
import logging
from functools import wraps
from datetime import datetime, timedelta
import json
from src.utils.logging_config import setup_logging, get_logger
from src.analysis.data_processor import DataProcessor
from src.analysis.data_storytelling import DataStorytelling
from src.analysis.metrics import BusinessMetrics

# Set up logging for API
logger = setup_logging('api')

# Create Blueprint
api = Blueprint('api', __name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set")
engine = create_engine(DATABASE_URL)

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    return wrapper

@api.route('/health')
@handle_errors
def health_check():
    """Health check endpoint"""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@api.route('/api/v1/products', methods=['GET'])
@handle_errors
def get_products():
    """Get products with optional filtering"""
    department = request.args.get('department')
    min_count = request.args.get('min_count', type=int)
    limit = request.args.get('limit', 100, type=int)
    
    query = """
    SELECT 
        p.product_id,
        p.product_name,
        d.department,
        COUNT(op.order_id) as order_count
    FROM products p
    JOIN departments d ON p.department_id = d.department_id
    LEFT JOIN order_products op ON p.product_id = op.product_id
    """
    
    where_clauses = []
    params = {}
    
    if department:
        where_clauses.append("d.department = :department")
        params['department'] = department
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " GROUP BY p.product_id, p.product_name, d.department"
    
    if min_count:
        query += " HAVING COUNT(op.order_id) >= :min_count"
        params['min_count'] = min_count
    
    query += " ORDER BY order_count DESC LIMIT :limit"
    params['limit'] = limit
    
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        products = [dict(row) for row in result]
    
    return jsonify({
        'status': 'success',
        'count': len(products),
        'data': products
    })

@api.route('/api/v1/orders', methods=['GET'])
@handle_errors
def get_orders():
    """Get orders with optional filtering"""
    user_id = request.args.get('user_id', type=int)
    limit = request.args.get('limit', 100, type=int)
    
    query = """
    SELECT 
        o.order_id,
        o.user_id,
        o.order_number,
        o.order_dow,
        o.order_hour_of_day,
        o.days_since_prior_order,
        COUNT(op.product_id) as product_count
    FROM orders o
    LEFT JOIN order_products op ON o.order_id = op.order_id
    """
    
    where_clauses = []
    params = {}
    
    if user_id:
        where_clauses.append("o.user_id = :user_id")
        params['user_id'] = user_id
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " GROUP BY o.order_id, o.user_id, o.order_number, o.order_dow, o.order_hour_of_day, o.days_since_prior_order"
    query += " ORDER BY o.order_id DESC LIMIT :limit"
    params['limit'] = limit
    
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        orders = [dict(row) for row in result]
    
    return jsonify({
        'status': 'success',
        'count': len(orders),
        'data': orders
    })

@api.route('/api/v1/analytics/department', methods=['GET'])
@handle_errors
def get_department_analytics():
    """Get department-level analytics"""
    query = """
    SELECT 
        d.department,
        COUNT(DISTINCT o.order_id) as order_count,
        COUNT(op.product_id) as product_count,
        COUNT(DISTINCT o.user_id) as unique_customers
    FROM departments d
    JOIN products p ON d.department_id = p.department_id
    JOIN order_products op ON p.product_id = op.product_id
    JOIN orders o ON op.order_id = o.order_id
    """
    query += " GROUP BY d.department ORDER BY order_count DESC"
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        analytics = [dict(row) for row in result]
    
    return jsonify({
        'status': 'success',
        'count': len(analytics),
        'data': analytics
    })

@api.route('/api/v1/analytics/customer/<int:user_id>', methods=['GET'])
@handle_errors
def get_customer_analytics(user_id):
    """Get customer-level analytics"""
    query = """
    WITH customer_orders AS (
        SELECT 
            o.user_id,
            o.order_id,
            COUNT(op.product_id) as product_count,
            COUNT(DISTINCT p.department_id) as department_count
        FROM orders o
        JOIN order_products op ON o.order_id = op.order_id
        JOIN products p ON op.product_id = p.product_id
        WHERE o.user_id = :user_id
        GROUP BY o.user_id, o.order_id
    )
    SELECT 
        user_id,
        COUNT(DISTINCT order_id) as total_orders,
        AVG(product_count) as avg_products_per_order,
        AVG(department_count) as avg_departments_per_order
    FROM customer_orders
    GROUP BY user_id
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {'user_id': user_id})
        analytics = dict(result.fetchone())
    
    return jsonify({
        'status': 'success',
        'data': analytics
    })

# Register the blueprint in app.py
# In app.py, add:
# from api import api
# app.server.register_blueprint(api) 