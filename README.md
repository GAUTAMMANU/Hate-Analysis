# Hate Speech Analysis Tool

A comprehensive tool for analyzing and visualizing hate speech in comments using the Gemini API and various visualization techniques.

## Features

- **Comment Analysis**
  - Batch processing of comments using Gemini API
  - Profanity pre-filtering using better-profanity
  - Rate limiting and retry mechanism
  - Partial results saving

- **Visualization**
  - Distribution of offensive vs non-offensive comments
  - Offense type breakdown
  - Severity score analysis
  - Top severe comments visualization
  - Comparison between different analysis results

- **Analysis Features**
  - Filter by offense type
  - Display top severe comments
  - Compare results between files
  - Generate detailed reports

## Security Notes

⚠️ **Important Security Considerations**:
- Never commit your `.env` file or API keys to version control
- Use environment variables for sensitive data
- Keep your API keys secure and rotate them regularly
- Be mindful of rate limits and API usage costs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hate-speech-analysis.git
cd hate-speech-analysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env
# Edit .env with your actual API key
```

## Usage

### Basic Analysis
```bash
python cli.py input.csv
```

### Visualization Options
```bash
# Generate all visualizations
python cli.py analyzed_comments.csv --charts

# Show top severe comments
python cli.py analyzed_comments.csv --top-severe 15

# Show top severe comments for specific type
python cli.py analyzed_comments.csv --top-severe 15 --filter-type "hate speech"

# Compare with another file
python cli.py analyzed_comments.csv --compare original.csv --compare-samples 500

# Filter by offense type
python cli.py analyzed_comments.csv --filter-type "toxicity"
```

### Command Line Arguments

- `input_file`: Path to the input CSV file (required)
- `--max-batches`: Maximum number of batches to process (default: 50)
- `--no-prefilter`: Disable profanity prefiltering
- `--charts`: Generate visualization charts
- `--compare`: Compare results with another analysis file
- `--compare-samples`: Number of samples to use for comparison
- `--top-severe`: Number of top severe comments to display
- `--filter-type`: Filter results by offense type
- `--output`: Output file path (default: analyzed_comments.csv)
- `--api-key`: Google API key (optional, can use environment variable)

### Offense Types
- hate speech
- toxicity
- profanity
- harassment

## Output Files

- `analyzed_comments.csv`: Main analysis results
- `partial_results.csv`: Partial results during analysis
- `offensive_distribution.png`: Distribution of offensive comments
- `offense_types.png`: Distribution of offense types
- `severity_distribution.png`: Severity score distribution
- `top_offensive_comments.png`: Top severe comments visualization
- `prefilter_comparison.png`: Comparison results (if --compare used)

## Error Handling

- Automatic retry mechanism for failed API requests
- User interaction for retry decisions
- Partial results saving during analysis
- Graceful handling of rate limits