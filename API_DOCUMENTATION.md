# 📡 API Documentation

## Overview

The `api.py` module provides RESTful API endpoints for the Instacart Market Basket Analysis platform. It's implemented as a Flask Blueprint and integrated with the main Dash application, enabling external systems to access data programmatically.

## 🎯 Benefits of API Integration

### 1. **External System Integration**
- **Mobile Applications**: Native mobile apps can fetch data and analytics
- **Business Intelligence Tools**: Connect Tableau, Power BI, or other BI platforms
- **Third-party Dashboards**: Embed analytics in external dashboards
- **CRM Systems**: Integrate customer analytics with customer relationship management

### 2. **Data Accessibility**
- **Programmatic Access**: Scripts and automation tools can fetch data
- **Real-time Analytics**: External systems can get live data updates
- **Data Export**: Automated data extraction for reporting
- **Webhook Integration**: Trigger actions based on data changes

### 3. **Scalability & Architecture**
- **Microservices Ready**: API-first architecture for future scaling
- **Multiple Frontends**: Support web, mobile, and desktop applications
- **API Gateway**: Centralized data access point
- **Caching Opportunities**: Implement caching for frequently accessed data

### 4. **Business Value**
- **Multi-platform Strategy**: Reach users across different platforms
- **Partner Integration**: Share data with business partners
- **Automated Reporting**: Generate reports without manual intervention
- **Data Monetization**: Potential to offer data as a service

## 🔗 Available Endpoints

### Products API
```http
GET /api/products
GET /api/products/{product_id}
GET /api/products/department/{department_id}
```

### Orders API
```http
GET /api/orders
GET /api/orders/{order_id}
GET /api/orders/user/{user_id}
```

### Analytics API
```http
GET /api/analytics/departments
GET /api/analytics/customers
GET /api/analytics/orders/summary
```

### Health Check
```http
GET /api/health
```

## 📱 Usage Examples

### 1. **Mobile App Integration**
```javascript
// Fetch top products for a department
fetch('/api/products/department/1')
  .then(response => response.json())
  .then(data => {
    console.log('Department products:', data);
  });

// Get customer analytics
fetch('/api/analytics/customers')
  .then(response => response.json())
  .then(data => {
    console.log('Customer insights:', data);
  });
```

### 2. **Python Script Integration**
```python
import requests
import pandas as pd

# Fetch all products
response = requests.get('http://localhost:8050/api/products')
products_df = pd.DataFrame(response.json())

# Get department analytics
response = requests.get('http://localhost:8050/api/analytics/departments')
dept_analytics = response.json()
```

### 3. **Business Intelligence Tool Connection**
```sql
-- Power BI or Tableau can connect via REST API
-- Example: DirectQuery to API endpoints
SELECT * FROM api_products
WHERE department_id = 1
```

### 4. **Automated Reporting**
```python
import requests
import schedule
import time

def generate_daily_report():
    # Fetch daily analytics
    response = requests.get('/api/analytics/orders/summary')
    data = response.json()
    
    # Send to email or save to file
    send_email_report(data)

# Schedule daily reports
schedule.every().day.at("09:00").do(generate_daily_report)
```

## 🛠️ Implementation Details

### Integration with Dash App
```python
# In app.py
from api import api
app.server.register_blueprint(api)
```

### Blueprint Structure
```python
# api.py creates a Flask Blueprint
api = Blueprint('api', __name__)

@api.route('/api/products')
def get_products():
    # Implementation
    pass
```

### Error Handling
- **404 Not Found**: Resource doesn't exist
- **500 Internal Error**: Server-side issues
- **JSON Responses**: Consistent error format

## 🔧 Configuration Options

### Environment Variables
```bash
# Database connection (shared with main app)
DATABASE_URL=postgresql://user:pass@host:port/db

# API-specific settings (future)
API_RATE_LIMIT=1000
API_CACHE_TTL=300
```

### CORS Support (Future Enhancement)
```python
# Enable cross-origin requests for web applications
from flask_cors import CORS
CORS(api)
```

## 📊 Response Formats

### Success Response
```json
{
  "status": "success",
  "data": [...],
  "timestamp": "2024-01-01T12:00:00Z",
  "count": 100
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Resource not found",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🚀 Future Enhancements

### 1. **Authentication & Authorization**
```python
# JWT token authentication
@api.route('/api/secure/products')
@jwt_required
def get_secure_products():
    pass
```

### 2. **Rate Limiting**
```python
# Prevent API abuse
@limiter.limit("1000 per hour")
@api.route('/api/products')
def get_products():
    pass
```

### 3. **Caching**
```python
# Redis caching for performance
@cache.cached(timeout=300)
@api.route('/api/analytics/departments')
def get_department_analytics():
    pass
```

### 4. **Webhook Support**
```python
# Notify external systems of data changes
@api.route('/api/webhooks/data-update', methods=['POST'])
def notify_data_update():
    pass
```

## 📈 Monitoring & Analytics

### API Usage Metrics
- Request count per endpoint
- Response times
- Error rates
- User agent tracking

### Health Monitoring
```bash
# Check API health
curl http://localhost:8050/api/health
```

## 🔒 Security Considerations

### Current Security
- Basic error handling
- Input validation
- SQL injection prevention

### Recommended Enhancements
- API key authentication
- Rate limiting
- Request logging
- HTTPS enforcement

## 📝 Migration Guide

### Removing API (If Not Needed)
1. **Remove from app.py:**
```python
# Remove these lines:
# from api import api
# app.server.register_blueprint(api)
```

2. **Delete api.py file**
3. **Update documentation**

### Adding New Endpoints
1. **Add route to api.py:**
```python
@api.route('/api/new-endpoint')
def new_endpoint():
    return jsonify({"data": "new data"})
```

2. **Update this documentation**
3. **Test with curl or Postman**

## 🎯 Use Cases by Industry

### E-commerce
- Inventory management integration
- Customer behavior analytics
- Order tracking systems

### Retail Analytics
- Store performance dashboards
- Product recommendation engines
- Customer segmentation tools

### Data Science
- Automated model training
- A/B testing platforms
- Real-time analytics pipelines

### Business Intelligence
- Executive dashboards
- Automated reporting
- KPI monitoring systems

---

## 📞 Support

For API-related questions or issues:
1. Check the logs in `logs/api.log`
2. Test endpoints with curl or Postman
3. Review this documentation
4. Check the main application logs for database issues

**Remember**: The API is optional but provides significant value for external integrations and future scalability! 