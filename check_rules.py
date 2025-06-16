from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM market_basket_rules'))
    rules_count = result.fetchone()[0]
    print(f'Rules stored in market_basket_rules: {rules_count}')
    
    result = conn.execute(text('SELECT COUNT(*) FROM market_basket_rules_weighted'))
    weighted_rules_count = result.fetchone()[0]
    print(f'Rules stored in market_basket_rules_weighted: {weighted_rules_count}')
    
    if rules_count > 0:
        print("\nSample rules:")
        result = conn.execute(text('SELECT support, confidence, lift, antecedents, consequents FROM market_basket_rules LIMIT 3'))
        for row in result:
            print(f"Support: {row[0]}, Confidence: {row[1]}, Lift: {row[2]}")
            print(f"  {row[3]} -> {row[4]}")
            print() 