import pandas as pd
import argparse
import logging
from collections import Counter
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('label_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def analyze_csv(csv_path, label_col='concept'):
    """
    Analyze label distribution in CSV file
    """
    try:
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        # Basic dataset statistics
        total_samples = len(df)
        unique_labels = df[label_col].unique()
        num_labels = len(unique_labels)
        
        logger.info("\nBasic Statistics:")
        logger.info(f"Total number of samples: {total_samples:,}")
        logger.info(f"Number of unique labels: {num_labels}")
        
        # Label distribution
        label_counts = df[label_col].value_counts()
        
        logger.info("\nLabel Distribution Statistics:")
        logger.info(f"Average samples per label: {label_counts.mean():.2f}")
        logger.info(f"Median samples per label: {label_counts.median():.2f}")
        logger.info(f"Min samples: {label_counts.min()} (Label: '{label_counts.index[-1]}')")
        logger.info(f"Max samples: {label_counts.max()} (Label: '{label_counts.index[0]}')")
        
        # Distribution metrics
        std_dev = label_counts.std()
        cv = std_dev / label_counts.mean()  # Coefficient of variation
        
        logger.info(f"\nDistribution Metrics:")
        logger.info(f"Standard deviation: {std_dev:.2f}")
        logger.info(f"Coefficient of variation: {cv:.2f}")
        
        # Detailed per-label analysis
        logger.info("\nDetailed Label Analysis:")
        print("\nLabel Distribution:")
        print("=" * 80)
        print(f"{'Label':<30} {'Count':>8} {'Percentage':>12} {'Example Attributes':<30}")
        print("-" * 80)
        
        for label in label_counts.index:
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            
            # Get example attributes
            examples = df[df[label_col] == label]['attribute_name'].head(2)
            examples_str = ', '.join(examples)
            
            print(f"{label[:30]:<30} {count:>8} {percentage:>11.2f}% {examples_str[:30]:<30}")
        
        # Check for class imbalance
        imbalance_ratio = label_counts.max() / label_counts.min()
        print("\nClass Balance Analysis:")
        print("=" * 80)
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("\nWARNING: Significant class imbalance detected!")
            print("Consider using one of these techniques:")
            print("1. Class weights in model training")
            print("2. Oversampling minority classes")
            print("3. Undersampling majority classes")
            print("4. Techniques like SMOTE for synthetic samples")
        
        # Save detailed statistics to CSV
        stats_df = pd.DataFrame({
            'Label': label_counts.index,
            'Count': label_counts.values,
            'Percentage': (label_counts.values / total_samples) * 100,
            'Cumulative_Percentage': (label_counts.cumsum() / total_samples) * 100
        })
        
        output_file = csv_path.rsplit('.', 1)[0] + '_label_stats.csv'
        stats_df.to_csv(output_file, index=False)
        logger.info(f"\nDetailed statistics saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error analyzing CSV: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Analyze label distribution in CSV file')
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('--label_col', default='concept', help='Name of label column (default: concept)')
    
    args = parser.parse_args()
    analyze_csv(args.csv_path, args.label_col)

if __name__ == "__main__":
    main()
