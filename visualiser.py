import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class HateSpeechVisualizer:
    def __init__(self, data_path: str, original_data_path: str = None):
        self.data = pd.read_csv(data_path)
        self.original_data = pd.read_csv(original_data_path) if original_data_path else None
        self.set_style()
        
    def set_style(self):
        sns.set_theme()
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        
    def compare_prefilter_results(self, n_samples: int = None):
        if self.original_data is None:
            raise ValueError("Original data path not provided for comparison")

        min_records = min(len(self.data), len(self.original_data))
        if n_samples is None or n_samples > min_records:
            n_samples = min_records

        merged = pd.merge(
            self.data[['comment_id', 'is_offensive', 'offense_type', 'severity']],
            self.original_data[['comment_id', 'is_offensive', 'offense_type', 'severity']],
            on='comment_id',
            suffixes=('_filtered', '_original')
        ).head(n_samples)

        # Confusion matrix
        confusion_matrix = pd.crosstab(
            merged['is_offensive_filtered'],
            merged['is_offensive_original'],
            rownames=['Pre-filtered'],
            colnames=['Original']
        )

        # Metrics
        tp = confusion_matrix.loc[True, True] if (True in confusion_matrix.index and True in confusion_matrix.columns) else 0
        fp = confusion_matrix.loc[True, False] if (True in confusion_matrix.index and False in confusion_matrix.columns) else 0
        fn = confusion_matrix.loc[False, True] if (False in confusion_matrix.index and True in confusion_matrix.columns) else 0
        tn = confusion_matrix.loc[False, False] if (False in confusion_matrix.index and False in confusion_matrix.columns) else 0

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix\n(Pre-filtered vs Original)')

        # Plot 2: Distribution Comparison
        comparison_counts = pd.DataFrame({
            'Pre-filtered': merged['is_offensive_filtered'].value_counts(),
            'Original': merged['is_offensive_original'].value_counts()
        })
        comparison_counts.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Offensive Content Distribution')
        axes[0, 1].set_xticklabels(['Non-Offensive', 'Offensive'], rotation=0)
        axes[0, 1].set_ylabel('Count')

        # Plot 3: Offense Type Comparison
        offense_type_counts = pd.DataFrame({
            'Pre-filtered': merged[merged['is_offensive_filtered']]['offense_type_filtered'].value_counts(),
            'Original': merged[merged['is_offensive_original']]['offense_type_original'].value_counts()
        })
        offense_type_counts.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Offense Type Distribution')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        severity_df = pd.DataFrame({
            'Pre-filtered': merged[merged['is_offensive_filtered']]['severity_filtered'],
            'Original': merged[merged['is_offensive_original']]['severity_original']
        })
        sns.boxplot(data=severity_df, ax=axes[1, 1])
        axes[1, 1].set_title('Severity Score Comparison')
        axes[1, 1].set_ylabel('Severity')

        plt.tight_layout()
        plt.savefig('prefilter_comparison.png')
        plt.close()

        # Print metrics
        print("\n=== Pre-filter Comparison Results ===")
        print(f"Sample Size: {n_samples}")
        print(f"Total Records in Filtered Data: {len(self.data)}")
        print(f"Total Records in Original Data: {len(self.original_data)}")
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        print("\nMetrics:")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives: {tn}")

        accuracy = (tp + tn) / n_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")


    def plot_offensive_distribution(self):
        plt.figure(figsize=(10, 8))
        offensive_counts = self.data['is_offensive'].value_counts()
        plt.pie(offensive_counts, 
                labels=['Non-Offensive', 'Offensive'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.1, 0))
        plt.title('Distribution of Offensive vs Non-Offensive Comments')
        plt.axis('equal')
        plt.savefig('offensive_distribution.png')
        plt.close()
        
    def plot_offense_types(self, offense_type: str = None):
        if offense_type:
            filtered_comments = self.data[self.data['offense_type'] == offense_type]
            if len(filtered_comments) == 0:
                print(f"No comments found with offense type: {offense_type}")
                return

            plt.figure(figsize=(12, 6))
            sns.histplot(data=filtered_comments, x='severity', bins=20)
            plt.title(f'Severity Distribution for {offense_type}')
            plt.xlabel('Severity Score')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f'severity_distribution_{offense_type.replace(" ", "_")}.png')
            plt.close()

            print(f"\nComments with offense type '{offense_type}':")
            print(f"Total count: {len(filtered_comments)}")
            print("-" * 80)
            for idx, row in filtered_comments.iterrows():
                print(f"\nComment ID: {row['comment_id']}")
                print(f"Username: {row['username']}")
                print(f"Comment: {row['original_comment']}")
                print(f"Severity: {row['severity']:.2f}")
                print(f"Explanation: {row['explanation']}")
                print("-" * 80)
        else:
            plt.figure(figsize=(12, 6))
            offense_counts = self.data['offense_type'].value_counts()
            sns.barplot(x=offense_counts.index, y=offense_counts.values)
            plt.title('Distribution of Offense Types')
            plt.xlabel('Offense Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('offense_types.png')
            plt.close()
        
    def plot_severity_distribution(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.data[self.data['is_offensive']], 
                    x='severity', 
                    bins=20,
                    kde=True)
        plt.title('Distribution of Severity Scores for Offensive Comments')
        plt.xlabel('Severity Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('severity_distribution.png')
        plt.close()
        
    def plot_offense_type_severity(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data[self.data['is_offensive']],
                   x='offense_type',
                   y='severity')
        plt.title('Severity Distribution by Offense Type')
        plt.xlabel('Offense Type')
        plt.ylabel('Severity Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('offense_type_severity.png')
        plt.close()
        
    def plot_top_offensive_comments(self, n: int = 10, offense_type: str = None):
        if offense_type:
            filtered_data = self.data[self.data['offense_type'] == offense_type]
            if len(filtered_data) == 0:
                print(f"No comments found with offense type: {offense_type}")
                return
            top_comments = filtered_data.nlargest(n, 'severity')
            print(f"\nTop {n} Most Severe Comments (Offense Type: {offense_type}):")
        else:
            top_comments = self.data.nlargest(n, 'severity')
            print(f"\nTop {n} Most Severe Comments:")

        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_comments,
                   x='severity',
                   y='original_comment',
                   orient='h')
        plt.title(f'Top {n} Most Severe Comments' + (f' ({offense_type})' if offense_type else ''))
        plt.xlabel('Severity Score')
        plt.ylabel('Comment')
        plt.tight_layout()
        plt.savefig('top_offensive_comments.png')
        plt.close()

        print("-" * 80)
        for idx, row in top_comments.iterrows():
            print(f"\nComment ID: {row['comment_id']}")
            print(f"Username: {row['username']}")
            print(f"Comment: {row['original_comment']}")
            print(f"Offense Type: {row['offense_type']}")
            print(f"Severity: {row['severity']:.2f}")
            print(f"Explanation: {row['explanation']}")
            print("-" * 80)

    def plot_offense_type_heatmap(self):
        plt.figure(figsize=(10, 8))
        offense_severity = self.data[self.data['is_offensive']].pivot_table(
            values='severity',
            index='offense_type',
            aggfunc=['mean', 'count']
        )
        sns.heatmap(offense_severity, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd')
        plt.title('Offense Type vs Severity Heatmap')
        plt.tight_layout()
        plt.savefig('offense_type_heatmap.png')
        plt.close()
        
    def generate_all_visualizations(self):
        print("Generating visualizations...")
        
        # Create standard visualizations
        self.plot_offensive_distribution()
        print("✓ Generated offensive distribution pie chart")
        
        self.plot_offense_types()
        print("✓ Generated offense types bar chart")
        
        self.plot_severity_distribution()
        print("✓ Generated severity distribution histogram")
        
        self.plot_offense_type_severity()
        print("✓ Generated offense type severity box plot")
        
        self.plot_top_offensive_comments()
        print("✓ Generated top offensive comments chart")
        
        self.plot_offense_type_heatmap()
        print("✓ Generated offense type heatmap")
        
        if self.original_data is not None:
            self.compare_prefilter_results()
            print("✓ Generated pre-filter comparison analysis")
        
        print("\nAll visualizations have been saved as PNG files in the current directory.")

if __name__ == "__main__":
    visualizer = HateSpeechVisualizer(
        "analyzed_comments.csv",  # Pre-filtered results
        "analyzed_comments_original.csv"  # Original results
    )
    visualizer.generate_all_visualizations()
