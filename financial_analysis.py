import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy import stats

# Load environment variables
load_dotenv()

# Database connection
engine = create_engine(os.getenv('DATABASE_URL'))

def load_rules_with_prices():
    """Load market basket rules and join with product prices"""
    with engine.connect() as conn:
        # Get rules with product information
        query = """
        SELECT 
            r.id,
            r.support,
            r.confidence,
            r.lift,
            r.antecedents,
            r.consequents,
            r.support_param,
            r.confidence_param,
            -- Extract product IDs from antecedents and consequents
            jsonb_array_elements_text(r.antecedents) as antecedent_product,
            jsonb_array_elements_text(r.consequents) as consequent_product
        FROM market_basket_rules r
        WHERE r.confidence >= 0.1
        ORDER BY r.confidence DESC
        LIMIT 1000
        """
        
        rules_df = pd.read_sql(query, conn)
        
        # Get product prices
        products_query = """
        SELECT product_id, product_name, price
        FROM products_with_price
        """
        products_df = pd.read_sql(products_query, conn)
        
        return rules_df, products_df

def calculate_rule_metrics(rules_df, products_df):
    """Calculate financial metrics for each rule using product_name for price lookup"""
    print("Calculating financial metrics for rules...")
    
    rule_metrics = []
    
    for _, rule in rules_df.iterrows():
        # Handle both string and list formats for antecedents/consequents
        if isinstance(rule['antecedents'], str):
            antecedents = json.loads(rule['antecedents'])
        else:
            antecedents = rule['antecedents']
            
        if isinstance(rule['consequents'], str):
            consequents = json.loads(rule['consequents'])
        else:
            consequents = rule['consequents']
        
        # Calculate average prices using product_name
        antecedent_prices = []
        consequent_prices = []
        
        for product_name in antecedents:
            product_price = products_df[products_df['product_name'] == product_name]['price'].iloc[0] if len(products_df[products_df['product_name'] == product_name]) > 0 else 0
            antecedent_prices.append(product_price)
        
        for product_name in consequents:
            product_price = products_df[products_df['product_name'] == product_name]['price'].iloc[0] if len(products_df[products_df['product_name'] == product_name]) > 0 else 0
            consequent_prices.append(product_price)
        
        avg_antecedent_price = np.mean(antecedent_prices) if antecedent_prices else 0
        avg_consequent_price = np.mean(consequent_prices) if consequent_prices else 0
        
        # Calculate financial metrics
        revenue_potential = rule['confidence'] * avg_consequent_price
        price_ratio = avg_consequent_price / avg_antecedent_price if avg_antecedent_price > 0 else 0
        aov_impact = revenue_potential - (rule['confidence'] * avg_antecedent_price)
        margin_opportunity = avg_consequent_price - avg_antecedent_price
        
        rule_metrics.append({
            'rule_id': rule['id'],
            'support': rule['support'],
            'confidence': rule['confidence'],
            'lift': rule['lift'],
            'antecedents': antecedents,
            'consequents': consequents,
            'avg_antecedent_price': avg_antecedent_price,
            'avg_consequent_price': avg_consequent_price,
            'revenue_potential': revenue_potential,
            'price_ratio': price_ratio,
            'aov_impact': aov_impact,
            'margin_opportunity': margin_opportunity
        })
    
    return pd.DataFrame(rule_metrics)

def show_specific_examples(metrics_df, similar_groups_df):
    """Show specific examples of rules with similar metrics but different consequents"""
    print("\n" + "="*80)
    print("SPECIFIC EXAMPLES: SIMILAR METRICS, DIFFERENT CONSEQUENTS")
    print("="*80)
    
    # Convert antecedents to tuples for grouping
    metrics_df['antecedents_tuple'] = metrics_df['antecedents'].apply(tuple)
    
    # Find rules with the same antecedents but different consequents
    print("\n1. RULES WITH SAME ANTECEDENTS, DIFFERENT CONSEQUENTS:")
    print("-" * 80)
    
    # Group by antecedents
    antecedent_groups = metrics_df.groupby('antecedents_tuple').agg({
        'consequents': list,
        'confidence': list,
        'avg_consequent_price': list,
        'revenue_potential': list,
        'aov_impact': list
    }).reset_index()
    
    # Find groups with multiple consequents
    multi_consequent_groups = antecedent_groups[antecedent_groups['consequents'].apply(len) > 1]
    
    if not multi_consequent_groups.empty:
        print(f"Found {len(multi_consequent_groups)} antecedent groups with multiple consequents")
        
        # Show top 3 examples
        for i, group in multi_consequent_groups.head(3).iterrows():
            antecedents = list(group['antecedents_tuple'])
            consequents = group['consequents']
            confidences = group['confidence']
            prices = group['avg_consequent_price']
            revenues = group['revenue_potential']
            aov_impacts = group['aov_impact']
            
            print(f"\nAntecedent: {antecedents}")
            print(f"{'Consequent':<40} {'Confidence':<12} {'Price':<8} {'Revenue':<10} {'AOV Impact':<12}")
            print("-" * 90)
            
            for j in range(len(consequents)):
                consequent = consequents[j]
                confidence = confidences[j]
                price = prices[j]
                revenue = revenues[j]
                aov = aov_impacts[j]
                
                print(f"{str(consequent):<40} {confidence:<12.3f} ${price:<7.2f} ${revenue:<9.2f} ${aov:<11.2f}")
            
            # Calculate differences
            if len(consequents) > 1:
                max_revenue_idx = revenues.index(max(revenues))
                min_revenue_idx = revenues.index(min(revenues))
                revenue_diff = revenues[max_revenue_idx] - revenues[min_revenue_idx]
                aov_diff = aov_impacts[max_revenue_idx] - aov_impacts[min_revenue_idx]
                
                print(f"\nRevenue difference: ${revenue_diff:.2f}")
                print(f"AOV impact difference: ${aov_diff:.2f}")
                print(f"Best choice for revenue: {consequents[max_revenue_idx]}")
                print(f"Best choice for AOV: {consequents[aov_impacts.index(max(aov_impacts))]}")
    else:
        print("No antecedent groups with multiple consequents found")
    
    # Show high price ratio examples
    print(f"\n2. HIGH PRICE RATIO EXAMPLES (UPSELL OPPORTUNITIES):")
    print("-" * 80)
    
    high_ratio_rules = metrics_df[metrics_df['price_ratio'] > 2.0].head(5)
    
    if not high_ratio_rules.empty:
        print(f"{'Antecedent':<30} {'Consequent':<30} {'Price Ratio':<12} {'Revenue':<10} {'AOV Impact':<12}")
        print("-" * 100)
        
        for _, rule in high_ratio_rules.iterrows():
            antecedent_str = str(rule['antecedents'])[:28] + "..." if len(str(rule['antecedents'])) > 30 else str(rule['antecedents'])
            consequent_str = str(rule['consequents'])[:28] + "..." if len(str(rule['consequents'])) > 30 else str(rule['consequents'])
            
            print(f"{antecedent_str:<30} {consequent_str:<30} {rule['price_ratio']:<12.2f} ${rule['revenue_potential']:<9.2f} ${rule['aov_impact']:<11.2f}")
    
    # Show some sample rules with their metrics
    print(f"\n3. SAMPLE RULES WITH THEIR METRICS:")
    print("-" * 80)
    
    sample_rules = metrics_df.head(5)
    print(f"{'Antecedent':<30} {'Consequent':<30} {'Confidence':<12} {'Price':<8} {'Revenue':<10} {'AOV':<8}")
    print("-" * 100)
    
    for _, rule in sample_rules.iterrows():
        antecedent_str = str(rule['antecedents'])[:28] + "..." if len(str(rule['antecedents'])) > 30 else str(rule['antecedents'])
        consequent_str = str(rule['consequents'])[:28] + "..." if len(str(rule['consequents'])) > 30 else str(rule['consequents'])
        
        print(f"{antecedent_str:<30} {consequent_str:<30} {rule['confidence']:<12.3f} ${rule['avg_consequent_price']:<7.2f} ${rule['revenue_potential']:<9.2f} ${rule['aov_impact']:<7.2f}")

def find_similar_rules(metrics_df, similarity_threshold=0.05):
    """Find rules with similar support, confidence, and lift but different prices"""
    print("Finding rules with similar metrics but different prices...")
    
    similar_groups = []
    
    # Group rules by similar metrics
    for i, rule1 in metrics_df.iterrows():
        for j, rule2 in metrics_df.iterrows():
            if i >= j:  # Avoid duplicate comparisons
                continue
                
            # Check if rules have similar metrics
            support_diff = abs(rule1['support'] - rule2['support'])
            confidence_diff = abs(rule1['confidence'] - rule2['confidence'])
            lift_diff = abs(rule1['lift'] - rule2['lift'])
            
            if (support_diff <= similarity_threshold and 
                confidence_diff <= similarity_threshold and 
                lift_diff <= similarity_threshold):
                
                # Check if they have different prices
                price_diff = abs(rule1['avg_consequent_price'] - rule2['avg_consequent_price'])
                revenue_diff = abs(rule1['revenue_potential'] - rule2['revenue_potential'])
                
                if price_diff > 0.5 or revenue_diff > 0.5:  # Significant price difference
                    similar_groups.append({
                        'rule1_id': rule1['rule_id'],
                        'rule2_id': rule2['rule_id'],
                        'rule1_confidence': rule1['confidence'],
                        'rule2_confidence': rule2['confidence'],
                        'rule1_price': rule1['avg_consequent_price'],
                        'rule2_price': rule2['avg_consequent_price'],
                        'rule1_revenue': rule1['revenue_potential'],
                        'rule2_revenue': rule2['revenue_potential'],
                        'rule1_aov': rule1['aov_impact'],
                        'rule2_aov': rule2['aov_impact'],
                        'price_difference': price_diff,
                        'revenue_difference': revenue_diff,
                        'aov_difference': abs(rule1['aov_impact'] - rule2['aov_impact']),
                        'rule1_antecedents': rule1['antecedents'],
                        'rule1_consequents': rule1['consequents'],
                        'rule2_antecedents': rule2['antecedents'],
                        'rule2_consequents': rule2['consequents']
                    })
    
    return pd.DataFrame(similar_groups)

def statistical_analysis(metrics_df, similar_groups_df):
    """Perform statistical analysis on the financial metrics"""
    print("Performing statistical analysis...")
    
    # 1. Correlation Analysis
    correlations = metrics_df[['confidence', 'lift', 'avg_consequent_price', 'revenue_potential', 'aov_impact']].corr()
    
    # 2. Price Distribution Analysis
    price_stats = {
        'mean_price': metrics_df['avg_consequent_price'].mean(),
        'median_price': metrics_df['avg_consequent_price'].median(),
        'std_price': metrics_df['avg_consequent_price'].std(),
        'price_skewness': stats.skew(metrics_df['avg_consequent_price'].dropna()),
        'price_kurtosis': stats.kurtosis(metrics_df['avg_consequent_price'].dropna())
    }
    
    # 3. Revenue vs Confidence Analysis
    revenue_confidence_corr = stats.pearsonr(metrics_df['confidence'], metrics_df['revenue_potential'])
    aov_confidence_corr = stats.pearsonr(metrics_df['confidence'], metrics_df['aov_impact'])
    
    # 4. Price Ratio Analysis
    price_ratio_stats = {
        'mean_ratio': metrics_df['price_ratio'].mean(),
        'median_ratio': metrics_df['price_ratio'].median(),
        'upsell_opportunities': len(metrics_df[metrics_df['price_ratio'] > 1.5]),
        'downsell_opportunities': len(metrics_df[metrics_df['price_ratio'] < 0.7])
    }
    
    # 5. Similar Rules Analysis
    if not similar_groups_df.empty:
        similar_stats = {
            'total_similar_pairs': len(similar_groups_df),
            'avg_price_difference': similar_groups_df['price_difference'].mean(),
            'avg_revenue_difference': similar_groups_df['revenue_difference'].mean(),
            'avg_aov_difference': similar_groups_df['aov_difference'].mean(),
            'high_value_opportunities': len(similar_groups_df[similar_groups_df['revenue_difference'] > 1.0])
        }
    else:
        similar_stats = {'total_similar_pairs': 0}
    
    return {
        'correlations': correlations,
        'price_stats': price_stats,
        'revenue_confidence_corr': revenue_confidence_corr,
        'aov_confidence_corr': aov_confidence_corr,
        'price_ratio_stats': price_ratio_stats,
        'similar_stats': similar_stats
    }

def generate_recommendation_insights(metrics_df, similar_groups_df, stats_results):
    """Generate business insights for recommendation strategies"""
    print("Generating recommendation insights...")
    
    insights = []
    
    # 1. Confidence vs Revenue Trade-off
    high_conf_low_revenue = metrics_df[
        (metrics_df['confidence'] > metrics_df['confidence'].quantile(0.75)) &
        (metrics_df['revenue_potential'] < metrics_df['revenue_potential'].quantile(0.5))
    ]
    
    low_conf_high_revenue = metrics_df[
        (metrics_df['confidence'] < metrics_df['confidence'].quantile(0.5)) &
        (metrics_df['revenue_potential'] > metrics_df['revenue_potential'].quantile(0.75))
    ]
    
    insights.append(f"High confidence, low revenue rules: {len(high_conf_low_revenue)}")
    insights.append(f"Low confidence, high revenue rules: {len(low_conf_high_revenue)}")
    
    # 2. Price Optimization Opportunities
    upsell_opportunities = metrics_df[metrics_df['price_ratio'] > 2.0]
    insights.append(f"High upsell opportunities (2x+ price ratio): {len(upsell_opportunities)}")
    
    # 3. AOV Impact Analysis
    high_aov_rules = metrics_df[metrics_df['aov_impact'] > metrics_df['aov_impact'].quantile(0.9)]
    insights.append(f"Top 10% AOV impact rules: {len(high_aov_rules)}")
    
    # 4. Similar Rules Trade-offs
    if not similar_groups_df.empty:
        best_revenue_gains = similar_groups_df.nlargest(5, 'revenue_difference')
        insights.append(f"Top 5 revenue improvement opportunities: ${best_revenue_gains['revenue_difference'].sum():.2f}")
    
    return insights

def print_financial_analysis(metrics_df, similar_groups_df, stats_results, insights):
    """Print comprehensive financial analysis"""
    print("\n" + "="*80)
    print("FINANCIAL ANALYSIS OF MARKET BASKET RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total rules analyzed: {len(metrics_df)}")
    print(f"Rules with similar metrics: {len(similar_groups_df)}")
    print(f"Average confidence: {metrics_df['confidence'].mean():.3f}")
    print(f"Average consequent price: ${metrics_df['avg_consequent_price'].mean():.2f}")
    
    print(f"\n2. PRICE DISTRIBUTION STATISTICS")
    print("-" * 40)
    price_stats = stats_results['price_stats']
    print(f"Mean price: ${price_stats['mean_price']:.2f}")
    print(f"Median price: ${price_stats['median_price']:.2f}")
    print(f"Price standard deviation: ${price_stats['std_price']:.2f}")
    print(f"Price skewness: {price_stats['price_skewness']:.3f}")
    print(f"Price kurtosis: {price_stats['price_kurtosis']:.3f}")
    
    print(f"\n3. CORRELATION ANALYSIS")
    print("-" * 40)
    print(f"Confidence vs Revenue correlation: {stats_results['revenue_confidence_corr'][0]:.3f} (p={stats_results['revenue_confidence_corr'][1]:.3f})")
    print(f"Confidence vs AOV correlation: {stats_results['aov_confidence_corr'][0]:.3f} (p={stats_results['aov_confidence_corr'][1]:.3f})")
    
    print(f"\n4. PRICE RATIO ANALYSIS")
    print("-" * 40)
    ratio_stats = stats_results['price_ratio_stats']
    print(f"Average price ratio: {ratio_stats['mean_ratio']:.2f}x")
    print(f"Median price ratio: {ratio_stats['median_ratio']:.2f}x")
    print(f"Upsell opportunities (1.5x+): {ratio_stats['upsell_opportunities']}")
    print(f"Downsell opportunities (<0.7x): {ratio_stats['downsell_opportunities']}")
    
    print(f"\n5. SIMILAR RULES ANALYSIS")
    print("-" * 40)
    similar_stats = stats_results['similar_stats']
    print(f"Total similar rule pairs: {similar_stats['total_similar_pairs']}")
    if similar_stats['total_similar_pairs'] > 0:
        print(f"Average price difference: ${similar_stats['avg_price_difference']:.2f}")
        print(f"Average revenue difference: ${similar_stats['avg_revenue_difference']:.2f}")
        print(f"Average AOV difference: ${similar_stats['avg_aov_difference']:.2f}")
        print(f"High-value opportunities: {similar_stats['high_value_opportunities']}")
    
    print(f"\n6. TOP 5 SIMILAR RULES WITH LARGEST REVENUE DIFFERENCES")
    print("-" * 60)
    if not similar_groups_df.empty:
        top_differences = similar_groups_df.nlargest(5, 'revenue_difference')
        print(f"{'Confidence':<10} {'Price1':<8} {'Price2':<8} {'Rev Diff':<10} {'AOV Diff':<10}")
        print("-" * 50)
        for _, row in top_differences.iterrows():
            print(f"{row['rule1_confidence']:<10.3f} ${row['rule1_price']:<7.2f} ${row['rule2_price']:<7.2f} ${row['revenue_difference']:<9.2f} ${row['aov_difference']:<9.2f}")
    
    print(f"\n7. BUSINESS INSIGHTS")
    print("-" * 40)
    for insight in insights:
        print(f"• {insight}")
    
    print(f"\n8. RECOMMENDATION STRATEGIES")
    print("-" * 40)
    print("• For maximum revenue: Prioritize rules with high price ratios and moderate confidence")
    print("• For AOV optimization: Focus on rules with high AOV impact regardless of confidence")
    print("• For risk mitigation: Choose high-confidence rules with stable price ratios")
    print("• For growth: Target low-confidence, high-revenue rules for testing")

def analyze_consequent_optimization(metrics_df):
    """Analyze rules with the same consequents but different antecedents for business optimization"""
    print("\n" + "="*80)
    print("CONSEQUENT OPTIMIZATION: SAME RECOMMENDATION, DIFFERENT TRIGGERS")
    print("="*80)
    
    # Convert consequents to tuples for grouping
    metrics_df['consequents_tuple'] = metrics_df['consequents'].apply(tuple)
    
    # Group by consequents
    consequent_groups = metrics_df.groupby('consequents_tuple').agg({
        'antecedents': list,
        'confidence': list,
        'support': list,
        'avg_antecedent_price': list,
        'avg_consequent_price': list,
        'revenue_potential': list,
        'aov_impact': list,
        'price_ratio': list
    }).reset_index()
    
    # Find groups with multiple antecedents
    multi_antecedent_groups = consequent_groups[consequent_groups['antecedents'].apply(len) > 1]
    
    print(f"Found {len(multi_antecedent_groups)} consequent groups with multiple antecedents")
    
    if not multi_antecedent_groups.empty:
        # Show top 5 examples
        for i, group in multi_antecedent_groups.head(5).iterrows():
            consequents = list(group['consequents_tuple'])
            antecedents_list = group['antecedents']
            confidences = group['confidence']
            supports = group['support']
            revenues = group['revenue_potential']
            aov_impacts = group['aov_impact']
            price_ratios = group['price_ratio']
            
            print(f"\nRecommendation: {consequents}")
            print(f"{'Antecedent':<50} {'Confidence':<12} {'Support':<10} {'Revenue':<10} {'AOV Impact':<12} {'Price Ratio':<12}")
            print("-" * 120)
            
            for j in range(len(antecedents_list)):
                antecedent = antecedents_list[j]
                confidence = confidences[j]
                support = supports[j]
                revenue = revenues[j]
                aov = aov_impacts[j]
                ratio = price_ratios[j]
                
                antecedent_str = str(antecedent)[:48] + "..." if len(str(antecedent)) > 50 else str(antecedent)
                print(f"{antecedent_str:<50} {confidence:<12.3f} {support:<10.6f} ${revenue:<9.2f} ${aov:<11.2f} {ratio:<12.2f}")
            
            # Calculate differences
            if len(antecedents_list) > 1:
                max_revenue_idx = revenues.index(max(revenues))
                min_revenue_idx = revenues.index(min(revenues))
                max_aov_idx = aov_impacts.index(max(aov_impacts))
                min_aov_idx = aov_impacts.index(min(aov_impacts))
                
                revenue_diff = revenues[max_revenue_idx] - revenues[min_revenue_idx]
                aov_diff = aov_impacts[max_aov_idx] - aov_impacts[min_aov_idx]
                confidence_diff = confidences[max_revenue_idx] - confidences[min_revenue_idx]
                
                print(f"\nBusiness Insights:")
                print(f"  Revenue difference: ${revenue_diff:.2f}")
                print(f"  AOV impact difference: ${aov_diff:.2f}")
                print(f"  Confidence difference: {confidence_diff:.3f}")
                print(f"  Best antecedent for revenue: {antecedents_list[max_revenue_idx]}")
                print(f"  Best antecedent for AOV: {antecedents_list[max_aov_idx]}")
                
                # Check if there's a trade-off
                if abs(confidence_diff) < 0.1 and revenue_diff > 0.5:
                    print(f"  ⭐ OPPORTUNITY: Similar confidence but ${revenue_diff:.2f} more revenue!")
                elif aov_diff > 1.0:
                    print(f"  ⭐ OPPORTUNITY: ${aov_diff:.2f} higher AOV impact!")
    
    # Find the most valuable consequent (highest average revenue)
    print(f"\n" + "="*80)
    print("MOST VALUABLE RECOMMENDATIONS")
    print("="*80)
    
    consequent_value = metrics_df.groupby('consequents_tuple').agg({
        'revenue_potential': 'mean',
        'aov_impact': 'mean',
        'confidence': 'mean',
        'avg_consequent_price': 'first'
    }).reset_index()
    
    consequent_value['consequents'] = consequent_value['consequents_tuple'].apply(list)
    top_consequents = consequent_value.nlargest(10, 'revenue_potential')
    
    print(f"{'Recommendation':<40} {'Avg Revenue':<12} {'Avg AOV':<12} {'Avg Confidence':<15} {'Price':<8}")
    print("-" * 100)
    
    for _, row in top_consequents.iterrows():
        consequent_str = str(row['consequents'])[:38] + "..." if len(str(row['consequents'])) > 40 else str(row['consequents'])
        print(f"{consequent_str:<40} ${row['revenue_potential']:<11.2f} ${row['aov_impact']:<11.2f} {row['confidence']:<15.3f} ${row['avg_consequent_price']:<7.2f}")

def main():
    """Main analysis function"""
    print("Starting Financial Analysis of Market Basket Recommendations...")
    
    # Load data
    rules_df, products_df = load_rules_with_prices()
    
    # Calculate metrics
    metrics_df = calculate_rule_metrics(rules_df, products_df)
    
    # Find similar rules
    similar_groups_df = find_similar_rules(metrics_df)
    
    # Statistical analysis
    stats_results = statistical_analysis(metrics_df, similar_groups_df)
    
    # Generate insights
    insights = generate_recommendation_insights(metrics_df, similar_groups_df, stats_results)
    
    # Print analysis
    print_financial_analysis(metrics_df, similar_groups_df, stats_results, insights)
    
    # Show specific examples
    show_specific_examples(metrics_df, similar_groups_df)
    
    # Analyze consequent optimization
    analyze_consequent_optimization(metrics_df)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 