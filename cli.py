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
    
    parser.add_argument('input_file', type=str, nargs='?', help='Path to the input CSV file')
    
    parser.add_argument('--max-batches', type=int, default=50,
                       help='Maximum number of batches to process (default: 50)')
    parser.add_argument('--no-prefilter', action='store_true',
                       help='Disable profanity prefiltering before API calls')
    parser.add_argument('--charts', action='store_true',
                       help='Generate visualization charts')
    parser.add_argument('--compare', type=str, metavar='ORIGINAL_FILE',
                       help='Compare results with original analysis file')
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

        # Check if only charts are needed
        if args.charts and not args.input_file:
            user_input = input("You have not specified an input file. Do you only need charts? (yes/no): ").lower()
            if user_input != 'yes':
                logging.error("Please provide an input file for analysis")
                return 1
            
            # Validate output file exists
            output_file = validate_file_path(args.output)
            
            # Initialize analyzer with output file for text output
            analyzer = CommentAnalyzer(data_path=output_file)
            analyzer.load_data()  # Load the analyzed data
            
            # Initialize visualizer with the output file
            visualizer = HateSpeechVisualizer(output_file)
            
            # Generate charts
            logging.info("Generating visualizations...")
            visualizer.generate_all_visualizations()

            if args.compare:
                compare_file = validate_file_path(args.compare)
                visualizer = HateSpeechVisualizer(output_file, compare_file)
                visualizer.compare_prefilter_results()

            # Handle filter-type and top-severe options
            if args.top_severe > 0:
                if args.filter_type:
                    logging.info(f"\nTop {args.top_severe} most severe comments for offense type: {args.filter_type}")
                    analyzer.display_top_severe_comments(args.top_severe, args.filter_type)
                    visualizer.plot_top_offensive_comments(args.top_severe, args.filter_type)
                else:
                    logging.info(f"\nTop {args.top_severe} most severe comments:")
                    analyzer.display_top_severe_comments(args.top_severe)
                    visualizer.plot_top_offensive_comments(args.top_severe)

            if args.filter_type and args.top_severe <= 0:
                logging.info(f"\nFiltered results for offense type: {args.filter_type}")
                analyzer.filter_by_offense_type(args.filter_type)
                visualizer.plot_offense_types(args.filter_type)
            
            return 0

        # Normal flow with input file
        input_file = validate_file_path(args.input_file)

        # Get API key from environment or argument
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided. Set it via --api-key or GOOGLE_API_KEY environment variable")

        # Initialize and run the analyzer
        analyzer = CommentAnalyzer(
            data_path=input_file,
            api_key=api_key,
            max_batches=args.max_batches,
            use_prefilter=not args.no_prefilter
        )

        logging.info("Loading and analyzing data...")
        analyzer.load_data()
        analyzer.analyze_all_comments()
        analyzer.generate_report()
        analyzer.save_results(args.output)
        logging.info(f"Analysis complete. Results saved to {args.output}")

        # Initialize visualizer with the analyzed data
        visualizer = HateSpeechVisualizer(args.output)

        if args.charts:
            logging.info("Generating visualizations...")
            visualizer.generate_all_visualizations()

            if args.compare:
                compare_file = validate_file_path(args.compare)
                visualizer = HateSpeechVisualizer(args.output, compare_file)
                visualizer.compare_prefilter_results()

        # Handle filter-type and top-severe options
        if args.top_severe > 0:
            if args.filter_type:
                logging.info(f"\nTop {args.top_severe} most severe comments for offense type: {args.filter_type}")
                analyzer.display_top_severe_comments(args.top_severe, args.filter_type)
                if args.charts:
                    visualizer.plot_top_offensive_comments(args.top_severe, args.filter_type)
            else:
                logging.info(f"\nTop {args.top_severe} most severe comments:")
                analyzer.display_top_severe_comments(args.top_severe)
                if args.charts:
                    visualizer.plot_top_offensive_comments(args.top_severe)

        if args.filter_type and args.top_severe <= 0:
            logging.info(f"\nFiltered results for offense type: {args.filter_type}")
            analyzer.filter_by_offense_type(args.filter_type)
            if args.charts:
                visualizer.plot_offense_types(args.filter_type)
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 