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
## Video demonstration: 
  - https://drive.google.com/file/d/1Y20vWAJFvPNWuhGrpYBRPWyVn-L2clc3/view
## Security Notes

⚠️ **Important Security Considerations**:
- Never commit your `.env` file or API keys to version control
- Use environment variables for sensitive data
- Keep your API keys secure and rotate them regularly
- Be mindful of rate limits and API usage costs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hate-analysis.git
cd hate-analysis
```

2. Create and activate a virtual environment:
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
cp .env.example .env
# Edit .env with your API key
```

## Usage

### Basic Analysis
```bash
python cli.py input.csv
```

### Advanced Options
```bash
# Generate visualizations only from existing analysis
python cli.py --charts

# Generate all visualizations with new analysis
python cli.py input.csv --charts

# Show top severe comments
python cli.py analyzed_comments.csv --top-severe 10

# Show top severe comments for specific type
python cli.py analyzed_comments.csv --top-severe 10 --filter-type "hate speech"

# Filter by offense type
python cli.py analyzed_comments.csv --filter-type "toxicity"

# Compare with another analysis file
python cli.py analyzed_comments.csv --compare original_analysis.csv

# Disable profanity prefiltering
python cli.py input.csv --no-prefilter

# Specify custom output file
python cli.py input.csv --output custom_results.csv

# Specify API key directly
python cli.py input.csv --api-key YOUR_API_KEY
```

### Command Line Arguments

- `input_file`: Path to the input CSV file (optional when only viewing charts)
- `--max-batches`: Maximum number of batches to process (default: 50)
- `--no-prefilter`: Disable profanity prefiltering before API calls
- `--charts`: Generate visualization charts
- `--compare`: Compare results with another analysis file
- `--top-severe`: Number of top severe comments to display (default: 10)
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

## API Request Management & Fail-Safe Mechanisms

### API Request Management
1. **Rate Limiting**
   - Automatic handling of API rate limits
   - Exponential backoff for retries
   - Configurable batch sizes

2. **Request Optimization**
   - Pre-filtering using local profanity detection
   - Batch processing to minimize API calls
   - Caching of analyzed results

### Fail-Safe Mechanisms
1. **Data Persistence**
   - Automatic saving of partial results after each batch
   - Recovery from previous analysis state
   - Checkpoint system for long-running analyses

2. **Error Recovery**
   - Automatic retry for failed API requests (3 attempts)
   - Interactive retry options for persistent failures
   - Graceful degradation with partial results
   - Detailed error logging and reporting

3. **Resource Management**
   - Memory-efficient batch processing
   - Disk space monitoring for output files
   - Cleanup of temporary files

## Error Handling

- Automatic retry mechanism for failed API requests
- User interaction for retry decisions
- Partial results saving during analysis
- Graceful handling of rate limits
