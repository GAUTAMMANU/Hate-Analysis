import argparse
import os
from load_data import CommentAnalyzer
from visualiser import HateSpeechVisualizer
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_file_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path

def main():
    parser = argparse.ArgumentParser(description='Hate Speech Analysis Tool')
    
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    
    parser.add_argument('--max-batches', type=int, default=50,
                       help='Maximum number of batches to process (default: 50)')
    parser.add_argument('--no-prefilter', action='store_true',
                       help='Disable profanity prefiltering before API calls')
    parser.add_argument('--charts', action='store_true',
                       help='Generate visualization charts')
    parser.add_argument('--compare', type=str, metavar='ORIGINAL_FILE',
                       help='Compare results with original analysis file')
    parser.add_argument('--compare-samples', type=int,
                       help='Number of samples to use for comparison (default: minimum of both files)')
    parser.add_argument('--top-severe', type=int, default=10,
                       help='Number of top severe comments to display (default: 10)')
    parser.add_argument('--filter-type', type=str, choices=['hate speech', 'toxicity', 'profanity', 'harassment'],
                       help='Filter results by offense type')
    parser.add_argument('--output', type=str, default='analyzed_comments.csv',
                       help='Output file path (default: analyzed_comments.csv)')
    parser.add_argument('--api-key', type=str, help='Google API key (optional, can use environment variable)')
    
    args = parser.parse_args()
    
    try:
        setup_logging()

        input_file = validate_file_path(args.input_file)

        visualizer = HateSpeechVisualizer(input_file)

        if args.charts:
            logging.info("Generating visualizations...")
            visualizer.generate_all_visualizations()

            if args.compare:
                compare_file = validate_file_path(args.compare)
                visualizer = HateSpeechVisualizer(input_file, compare_file)

                n_samples = args.compare_samples
                if n_samples is not None:
                    logging.info(f"Using {n_samples} samples for comparison")
                else:
                    logging.info("Using minimum number of records from both files for comparison")
                
                visualizer.compare_prefilter_results(n_samples)
        

        if args.top_severe > 0:
            if args.filter_type:
                logging.info(f"\nTop {args.top_severe} most severe comments for offense type: {args.filter_type}")
                visualizer.plot_top_offensive_comments(args.top_severe, args.filter_type)
            else:
                logging.info(f"\nTop {args.top_severe} most severe comments:")
                visualizer.plot_top_offensive_comments(args.top_severe)

        if args.filter_type and args.top_severe <= 0:
            logging.info(f"\nFiltered results for offense type: {args.filter_type}")
            visualizer.plot_offense_types(args.filter_type)
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 