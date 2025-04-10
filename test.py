import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import time
from datetime import datetime

# Load .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Load dataset
df = pd.read_csv("labeled_data.csv")
df.rename(columns={"tweet": "comment_text"}, inplace=True)

# Choose how many test comments to send
NUM_COMMENTS_TO_TEST = 15  # This will process 150 comments (15 batches of 10)
BATCH_SIZE = 10

# Prompt template for batch processing
def build_batch_prompt(comments):
    comments_text = "\n".join([f"{i+1}. '{comment}'" for i, comment in enumerate(comments)])
    return f"""Analyze the following comments for offensive content. For each comment, classify it into one of these categories: hate speech, toxicity, profanity, harassment.

Comments:
{comments_text}

For each comment, provide a JSON object with these fields:
- is_offensive (boolean)
- offense_type (string, must be one of: hate speech, toxicity, profanity, harassment, or none)
- explanation (string)
- severity (float between 0 and 1, where 1 is most severe)

Return the results as a JSON array of objects, one for each comment in the same order.
"""

def process_batch(comments):
    """Process a batch of comments and return the results."""
    try:
        prompt = build_batch_prompt(comments)
        response = model.generate_content(prompt)
        
        # Clean and parse the response
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        results = json.loads(cleaned_text)
        return results
    except Exception as e:
        print(f"❌ Error processing batch: {e}")
        return [{
            "is_offensive": False,
            "offense_type": "error",
            "explanation": f"Error in analysis: {str(e)}",
            "severity": 0.0
        } for _ in range(len(comments))]

# Process comments in batches
total_comments = NUM_COMMENTS_TO_TEST * BATCH_SIZE
print(f"\n🔹 Processing {total_comments} comments in {NUM_COMMENTS_TO_TEST} batches of {BATCH_SIZE}...\n")

all_results = []
for batch_num in range(NUM_COMMENTS_TO_TEST):
    start_idx = batch_num * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    batch_comments = df["comment_text"].iloc[start_idx:end_idx].tolist()
    
    print(f"\n📦 Processing Batch {batch_num + 1}/{NUM_COMMENTS_TO_TEST}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    batch_results = process_batch(batch_comments)
    all_results.extend(batch_results)
    
    # Print results for this batch
    for i, (comment, result) in enumerate(zip(batch_comments, batch_results)):
        print(f"\n🔹 Comment {start_idx + i + 1}:")
        print(f"   Text: {comment}")
        print(f"   Offensive: {result['is_offensive']}")
        print(f"   Type: {result['offense_type']}")
        print(f"   Severity: {result['severity']:.2f}")
        print(f"   Explanation: {result['explanation']}")
    
    # Add delay between batches to respect rate limits
    if batch_num < NUM_COMMENTS_TO_TEST - 1:
        print("\n⏳ Waiting 5 seconds before next batch...")
        time.sleep(5)

# Print summary
print("\n📊 Summary:")
print(f"Total Comments Processed: {len(all_results)}")
offensive_count = sum(1 for r in all_results if r['is_offensive'])
print(f"Offensive Comments: {offensive_count}")

# Save results to file
output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n💾 Results saved to {output_file}")
