#!/usr/bin/env python3
"""
Show all calculated results from the retail customer analytics pipeline.
This includes all KPIs, charts, and key tables that appear in the UI.
"""

import pandas as pd
import tempfile
import os
import json
import sys

def main():
    # Load sample data
    df = pd.read_parquet('sample_data.parquet')
    
    # Map columns to pipeline expected format
    mapped = pd.DataFrame({
        'customer_id': df['customer_id'].astype('string'),
        'transaction_date': pd.to_datetime(df['transaction_datetime']),
        'transaction_id': df['invoice_id'].astype('string'),
        'product_id': df['product_id'].astype('string'),
        'product_description': df['description'].astype('string'),
        'department': df['department'].astype('string'),
        'category': df['category'].astype('string'),
        'price': df['unit_price'].astype(float),
        'quantity': df['quantity'].astype(int)
    })
    mapped = mapped.dropna(subset=['customer_id'])
    
    print(f"Input data: {len(mapped):,} rows, {mapped['customer_id'].nunique():,} customers")
    print(f"Date range: {mapped['transaction_date'].min()} to {mapped['transaction_date'].max()}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        mapped.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        import retail_customer_analysis as rca
        import insight_engine as ie
        
        with tempfile.TemporaryDirectory() as tmpdir:
            res = f'{tmpdir}/results'
            out = f'{tmpdir}/insights'
            
            # Run pipeline
            code = rca.main([
                '--input', temp_path, '--output-dir', res,
                '--horizon-days', '90', '--windows-days', '30,90,365',
                '--min-clusters', '3', '--max-clusters', '7'
            ])
            
            if code != 0:
                print("Pipeline failed!")
                return
            
            ie_code = ie.main(['--results-dir', res, '--output-dir', out, '--dpi', '110', '--top-n', '12'])
            
            # Load insight summary
            with open(f'{out}/insight_summary.json') as f:
                summary = json.load(f)
            
            # ========== KPIs ==========
            print("=" * 80)
            print("KPIs (from insight_summary.json)")
            print("=" * 80)
            kpis = summary.get('kpis', [])
            for item in kpis:
                if isinstance(item, dict):
                    label = item.get('label', item.get('key', 'N/A'))
                    value = item.get('value', item.get('v', 'N/A'))
                    print(f"  {label}: {value}")
                else:
                    print(f"  {item}")
            
            # ========== CHARTS ==========
            print("\n" + "=" * 80)
            print("CHARTS GENERATED (from insight_summary.json)")
            print("=" * 80)
            charts = summary.get('charts', [])
            for c in charts:
                print(f"  [{c['key']}] {c['title']}")
                print(f"    → {c['takeaway']}")
            
            # ========== KEY TABLES ==========
            print("\n" + "=" * 80)
            print("KEY TABLES SUMMARY")
            print("=" * 80)
            
            # Customer features
            cf = pd.read_csv(f'{res}/customer_features.csv')
            print(f"\ncustomer_features: {cf.shape[0]:,} rows, {cf.shape[1]} cols")
            if 'p_alive' in cf.columns:
                alive = (cf['p_alive'] >= 0.5).sum()
                print(f"  P(alive) >= 0.5: {alive:,} / {len(cf):,} ({alive/len(cf)*100:.1f}%)")
                print(f"  P(alive) stats: mean={cf['p_alive'].mean():.3f}, median={cf['p_alive'].median():.3f}")
            if 'expected_revenue_horizon' in cf.columns:
                print(f"  Expected revenue (horizon): {cf['expected_revenue_horizon'].sum():,.2f}")
            if 'cluster_label' in cf.columns:
                print(f"  Clusters: {cf['cluster_label'].nunique()}")
                for cluster, count in cf['cluster_label'].value_counts().items():
                    print(f"  {cluster}: {count:,} ({count/len(cf)*100:.1f}%)")
            
            # Holdout metrics
            hm = pd.read_csv(f'{res}/holdout_metrics.csv')
            print(f"\nholdout_metrics:")
            print(hm[['model', 'target', 'auc_any_purchase', 'mae', 'spearman']].to_string(index=False))
            
            # Calibration
            cal = pd.read_csv(f'{res}/holdout_calibration.csv')
            print(f"\nholdout_calibration (deciles):")
            print(cal.to_string(index=False))
            
            # Cohort retention
            cr = pd.read_csv(f'{res}/cohort_retention_new_customers.csv')
            m1 = cr[(cr['months_since_cohort'] == 1) & (cr['period_complete'] == True)]
            print(f"\ncohort_retention (Month 1, complete periods):")
            print(m1[['cohort_month', 'cohort_size', 'retention_rate']].to_string(index=False))
            
            # Cluster diagnostics
            cd = pd.read_csv(f'{res}/cluster_diagnostics.csv')
            print(f"\ncluster_diagnostics:")
            print(cd[['candidate_k', 'silhouette', 'stability_ari_mean', 'selected']].to_string(index=False))
            
            # Repeat survival
            rs = pd.read_csv(f'{res}/repeat_survival.csv')
            s90 = rs[rs['day'] == 90]
            print(f"\nrepeat_survival (90 days):")
            print(s90[['group', 'customers', 'repeat_probability']].to_string(index=False))
            
            # Revenue concentration
            conc = pd.read_csv(f'{res}/customer_revenue_concentration.csv')
            print(f"\ncustomer_revenue_concentration:")
            print(conc.to_string(index=False))
            
            # Category summary
            cat = pd.read_csv(f'{res}/category_summary.csv')
            print(f"\ncategory_summary (top 10 by revenue):")
            print(cat.nlargest(10, 'gross_purchase_revenue')[['category', 'gross_purchase_revenue', 'return_value']].to_string(index=False))
            
            # Department summary
            dept = pd.read_csv(f'{res}/department_summary.csv')
            print(f"\ndepartment_summary:")
            print(dept[['department', 'gross_purchase_revenue', 'return_value']].to_string(index=False))
            
            # Product summary
            prod = pd.read_csv(f'{res}/product_summary.csv')
            print(f"\nproduct_summary (top 10):")
            cols = [c for c in ['product_id', 'product_description', 'gross_purchase_revenue'] if c in prod.columns]
            print(prod.nlargest(10, 'gross_purchase_revenue')[cols].to_string(index=False))
            
            # Basket affinity
            ba = pd.read_csv(f'{res}/basket_affinity.csv')
            if 'passes_all_filters' in ba.columns:
                passed = ba[ba['passes_all_filters'].astype(bool)]
                print(f"\nbasket_affinity: {len(passed):,} / {len(ba):,} rules pass all filters")
                if len(passed) > 0:
                    cols = [c for c in ['antecedent', 'consequent', 'customer_lift', 'confidence'] if c in passed.columns]
                    print(passed.nlargest(5, 'customer_lift')[cols].to_string(index=False))
            
            # Next trip affinity
            nta = pd.read_csv(f'{res}/next_trip_affinity.csv')
            if len(nta) > 0:
                print(f"\nnext_trip_affinity (top 5):")
                cols = [c for c in ['from_item', 'next_trip_item', 'next_trip_lift', 'p_next_given_from'] if c in nta.columns]
                print(nta.nlargest(5, 'next_trip_lift')[cols].to_string(index=False))
            
            # Monthly summary
            ms = pd.read_csv(f'{res}/monthly_summary.csv')
            if len(ms) > 0:
                print(f"\nmonthly_summary:")
                print(ms.to_string(index=False))
            
            # Cluster profiles
            cp = pd.read_csv(f'{res}/cluster_profiles.csv')
            if len(cp) > 0:
                print(f"\ncluster_profiles:")
                print(cp.to_string(index=False))
            
            # RFM segment profiles
            rfm = pd.read_csv(f'{res}/rfm_segment_profiles.csv')
            if len(rfm) > 0:
                print(f"\nrfm_segment_profiles:")
                print(rfm.to_string(index=False))
            
            print("\n" + "=" * 80)
            print("DONE - All results displayed above")
            print("=" * 80)
            
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    main()