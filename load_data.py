import pandas as pd
import os
import re
from typing import Dict, List
import json
import logging
import time
from datetime import datetime, timedelta
import google.generativeai as genai
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
from better_profanity import profanity

# Load environment variables from .env file
load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

profanity.load_censor_words()

class RateLimit:
    def __init__(self):
        self.requests_today = 0
        self.last_request_time = None
        self.requests_this_minute = 0
        self.day_start = datetime.now()

    def can_make_request(self) -> bool:
        current_time = datetime.now()
        
        # Reset daily counter if it's a new day
        if current_time.date() != self.day_start.date():
            self.requests_today = 0
            self.day_start = current_time

        if self.requests_today >= 50:  
            return False

        if self.last_request_time and (current_time - self.last_request_time) >= timedelta(minutes=1):
            self.requests_this_minute = 0

        if self.requests_this_minute >= 15:
            return False

        return True

    def record_request(self):
        current_time = datetime.now()
        self.requests_today += 1
        self.requests_this_minute += 1
        self.last_request_time = current_time

class CommentAnalyzer:
    def __init__(self, data_path: str, api_key: str = None, max_batches: int = 50, use_prefilter: bool = True):
        self.data_path = data_path
        self.data = None
        self.analyzed_data = None
        self.rate_limiter = RateLimit()
        self.batch_size = 20  
        self.max_batches = max_batches  
        self.use_prefilter = use_prefilter  
        
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            raise ValueError("Gemini API key not provided")

    def pre_filter_comments(self, comments: List[str]) -> List[str]:
        return [comment for comment in comments if profanity.contains_profanity(comment)]

    def load_data(self) -> None:
        try:
            self.data = pd.read_csv(self.data_path)
            self.data.rename(columns={"tweet": "comment_text"}, inplace=True)
            logging.info(f"Successfully loaded data with {len(self.data)} comments")
            
            # Calculate maximum comments we can process based on max_batches
            max_comments = self.batch_size * self.max_batches
            if len(self.data) > max_comments:
                logging.warning(f"Dataset contains {len(self.data)} comments, but we can only process {max_comments} comments ({self.max_batches} batches × {self.batch_size} comments per batch).")
                logging.warning(f"Only the first {max_comments} comments will be processed.")
                self.data = self.data.head(max_comments)
            
            logging.info("\nData Preview:")
            print(self.data.head())
            logging.info("\nData Summary:")
            print(self.data.info())
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def build_batch_prompt(self, comments: List[str]) -> str:
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

    def process_batch(self, comments: List[str]) -> List[Dict]:
        if self.use_prefilter:
            # Pre-filter comments
            potentially_offensive = self.pre_filter_comments(comments)
            
            if not potentially_offensive:
                # If no potentially offensive comments, return default results
                return [{
                    "is_offensive": False,
                    "offense_type": "none",
                    "explanation": "No profanity detected by pre-filter",
                    "severity": 0.0
                } for _ in range(len(comments))]
        else:
            potentially_offensive = comments

        max_retries = 3
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                while not self.rate_limiter.can_make_request():
                    time.sleep(2)  

                prompt = self.build_batch_prompt(potentially_offensive)
                response = self.model.generate_content(prompt)
                self.rate_limiter.record_request()

                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]

                results = json.loads(cleaned_text)
                
                final_results = []
                result_idx = 0
                
                for comment in comments:
                    if comment in potentially_offensive:
                        final_results.append(results[result_idx])
                        result_idx += 1
                    else:
                        final_results.append({
                            "is_offensive": False,
                            "offense_type": "none",
                            "explanation": "No profanity detected by pre-filter" if self.use_prefilter else "Analyzed without pre-filter",
                            "severity": 0.0
                        })
                
                return final_results

            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logging.error(f"Error processing batch (attempt {retry_count}/{max_retries}): {last_error}")
                
                if retry_count < max_retries:
                    # Ask user if they want to retry
                    user_input = input(f"Request failed. Retry? (y/n) [Attempt {retry_count}/{max_retries}]: ").lower()
                    if user_input != 'y':
                        logging.info("User chose to stop retrying. Saving partial results...")
                        break
                    logging.info(f"Retrying request (attempt {retry_count + 1}/{max_retries})...")
                    time.sleep(5) 

        logging.error(f"Failed to process batch after {retry_count} attempts. Last error: {last_error}")
        return [{
            "is_offensive": False,
            "offense_type": "error",
            "explanation": f"Error in analysis after {retry_count} attempts: {last_error}",
            "severity": 0.0
        } for _ in range(len(comments))]

    def analyze_all_comments(self) -> None:
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        results = []
        total_comments = len(self.data)
        num_batches = min(self.max_batches, (total_comments + self.batch_size - 1) // self.batch_size)
        max_comments = num_batches * self.batch_size

        logging.info(f"\nStarting analysis with {num_batches} batches ({max_comments} comments)")
        
        for batch_num in range(num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_comments)
            batch_comments = self.data['comment_text'].iloc[start_idx:end_idx].tolist()
            
            logging.info(f"\nProcessing Batch {batch_num + 1}/{num_batches}")
            logging.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            logging.info(f"Processing comments {start_idx + 1} to {end_idx} of {max_comments}")
            
            try:
                batch_results = self.process_batch(batch_comments)
                
                for i, result in enumerate(batch_results):
                    result['comment_id'] = start_idx + i
                    result['username'] = f"user_{start_idx + i}"
                    result['original_comment'] = batch_comments[i]
                
                results.extend(batch_results)
                
                self.analyzed_data = pd.DataFrame(results)
                self.save_results("partial_results.csv")
                logging.info(f"Saved {len(results)} results so far")
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_num + 1}: {str(e)}")
                logging.info("Saving partial results before continuing...")
                self.analyzed_data = pd.DataFrame(results)
                self.save_results("partial_results.csv")
                logging.info(f"Saved {len(results)} results before error")
                continue
            
            if batch_num < num_batches - 1:
                logging.info("Waiting 5 seconds before next batch...")
                time.sleep(5)

        self.analyzed_data = pd.DataFrame(results)
        logging.info(f"Analysis completed. Processed {len(results)} comments in {num_batches} batches.")

    def generate_report(self) -> None:
        if self.analyzed_data is None:
            raise ValueError("No analyzed data available. Run analyze_all_comments() first.")

        total_comments = len(self.analyzed_data)
        offensive_comments = self.analyzed_data[self.analyzed_data['is_offensive'] == True]
        
        print("\n=== Analysis Report ===")
        print(f"Total Comments Analyzed: {total_comments}")
        print(f"Offensive Comments: {len(offensive_comments)}")
        
        print("\nOffense Type Breakdown:")
        offense_types = offensive_comments['offense_type'].value_counts()
        for offense_type, count in offense_types.items():
            print(f"- {offense_type}: {count}")

    def save_results(self, output_path: str) -> None:
        if self.analyzed_data is None:
            raise ValueError("No analyzed data available. Run analyze_all_comments() first.")

        self.analyzed_data.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")

    # def display_top_severe_comments(self, n: int = 10, offense_type: str = None) -> None:
    #     if self.analyzed_data is None:
    #         raise ValueError("No analyzed data available. Run analyze_all_comments() first.")

    #     if offense_type:
    #         filtered_data = self.analyzed_data[self.analyzed_data['offense_type'] == offense_type]
    #         if len(filtered_data) == 0:
    #             print(f"No comments found with offense type: {offense_type}")
    #             return
    #         top_comments = filtered_data.nlargest(n, 'severity')
    #         print(f"\nTop {n} Most Severe Comments (Offense Type: {offense_type}):")
    #     else:
    #         top_comments = self.analyzed_data.nlargest(n, 'severity')
    #         print(f"\nTop {n} Most Severe Comments:")

    #     print("-" * 80)
    #     for idx, row in top_comments.iterrows():
    #         print(f"\nComment ID: {row['comment_id']}")
    #         print(f"Username: {row['username']}")
    #         print(f"Comment: {row['original_comment']}")
    #         print(f"Offense Type: {row['offense_type']}")
    #         print(f"Severity: {row['severity']:.2f}")
    #         print(f"Explanation: {row['explanation']}")
    #         print("-" * 80)

    def filter_by_offense_type(self, offense_type: str) -> None:
        if self.analyzed_data is None:
            raise ValueError("No analyzed data available. Run analyze_all_comments() first.")

        filtered_comments = self.analyzed_data[self.analyzed_data['offense_type'] == offense_type]
        
        if len(filtered_comments) == 0:
            print(f"No comments found with offense type: {offense_type}")
            return

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

if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key not found! Please create a .env file with your GOOGLE_API_KEY or set it as an environment variable."
        )
    
    # Example usage
    analyzer = CommentAnalyzer(
        data_path="labeled_data.csv",
        api_key=api_key
    )
    
    analyzer.load_data()
    analyzer.analyze_all_comments()
    analyzer.generate_report()
    analyzer.save_results("analyzed_comments.csv")
