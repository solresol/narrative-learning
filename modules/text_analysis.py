#!/usr/bin/env python3
import re
from collections import Counter
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for linguistic analysis.
    
    Args:
        text: Raw text string
        
    Returns:
        List of tokens (words)
    """
    # Convert to lowercase and split into words
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Split into words and filter out empty strings
    words = [word for word in text.split() if word]
    return words

def get_word_frequencies(text_list: List[str]) -> Counter:
    """
    Calculate word frequencies from a list of texts.
    
    Args:
        text_list: List of text strings
        
    Returns:
        Counter object with word frequencies
    """
    all_words = []
    for text in text_list:
        all_words.extend(preprocess_text(text))
    
    return Counter(all_words)

def calculate_zipfs_law(text_list: List[str]) -> Dict[str, Union[float, dict]]:
    """
    Calculate Zipf's law coefficient from a list of texts.
    Zipf's law states that the frequency of a word is inversely proportional to its rank.
    
    This implementation uses a fixed seed for random sampling to ensure deterministic results
    while still sampling 1000 words with replacement 5 times and averaging the results
    to handle documents of different lengths.
    
    Args:
        text_list: List of text strings
        
    Returns:
        Dictionary with Zipf coefficient and related statistics
    """
    if not text_list:
        return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
    
    # Preprocess all text and get a list of all words
    all_words = []
    for text in text_list:
        all_words.extend(preprocess_text(text))
        
    if not all_words:
        return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
    
    # Number of runs, sample size and results storage
    num_runs = 5
    sample_size = 1000
    zipf_coefficients = []
    r_squared_values = []
    
    # Set fixed seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Run multiple times and average the results
    for run in range(num_runs):
        # Sample words with replacement
        if len(all_words) == 0:
            continue
            
        # Sample with replacement using seeded RNG
        sampled_words = rng.choice(all_words, size=sample_size, replace=True)
        
        # Count word frequencies in the sample
        word_counts = Counter(sampled_words)
        if not word_counts:
            continue
            
        # Get frequency and rank
        frequencies = []
        ranks = []
        
        sorted_items = word_counts.most_common()
        for rank, (word, count) in enumerate(sorted_items, 1):
            frequencies.append(count)
            ranks.append(rank)
        
        # Convert to numpy arrays and calculate log values
        log_ranks = np.log(ranks)
        log_frequencies = np.log(frequencies)
        
        # Linear regression to find Zipf coefficient
        # Zipf's law: frequency ∝ 1/rank^α where α is the Zipf coefficient
        # In log space: log(frequency) = -α * log(rank) + constant
        slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
        zipf_coefficient = -slope  # The slope is negative, so we negate it
        
        # Calculate R-squared
        y_pred = slope * log_ranks + intercept
        ss_total = np.sum((log_frequencies - np.mean(log_frequencies))**2)
        ss_residual = np.sum((log_frequencies - y_pred)**2)
        if ss_total == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_residual / ss_total)
        
        zipf_coefficients.append(zipf_coefficient)
        r_squared_values.append(r_squared)
    
    # Calculate average values
    if not zipf_coefficients:
        return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
        
    avg_zipf_coefficient = sum(zipf_coefficients) / len(zipf_coefficients)
    avg_r_squared = sum(r_squared_values) / len(r_squared_values)
    
    # Return the average values and some additional information
    return {
        'coefficient': avg_zipf_coefficient,
        'r_squared': avg_r_squared,
        'data': {
            'individual_coefficients': zipf_coefficients,
            'individual_r_squared': r_squared_values
        }
    }

def calculate_herdans_law(text_list: List[str]) -> Dict[str, Union[float, dict]]:
    """
    Calculate Herdan's law (Heaps' law) coefficient from a list of texts.
    Herdan's law describes the relationship between vocabulary size and text length.
    
    This implementation uses a fixed seed for random sampling to ensure deterministic results
    while still sampling 1000 words with replacement 5 times and averaging the results
    to handle documents of different lengths.
    
    Args:
        text_list: List of text strings
        
    Returns:
        Dictionary with Herdan coefficient and related statistics
    """
    if not text_list:
        return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
        
    # Preprocess all text and get a list of all words
    all_words = []
    for text in text_list:
        all_words.extend(preprocess_text(text))
        
    if not all_words:
        return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
        
    # Number of runs, sample size and results storage
    num_runs = 5
    sample_size = 1000
    herdan_coefficients = []
    r_squared_values = []
    
    # Set fixed seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Run multiple times and average the results
    for run in range(num_runs):
        if len(all_words) == 0:
            continue
        
        # Sample with replacement using seeded RNG
        sampled_words = list(rng.choice(all_words, size=sample_size, replace=True))
        
        # Calculate vocabulary growth as we read through the sample
        vocab_sizes = []
        text_lengths = []
        
        seen_words = set()
        for i, word in enumerate(sampled_words, 1):
            seen_words.add(word)
            
            # Add a data point for every 10 words or so to get enough data points
            if i % 10 == 0 or i == len(sampled_words):
                vocab_sizes.append(len(seen_words))
                text_lengths.append(i)
        
        if len(vocab_sizes) < 3:  # Need at least 3 points for a meaningful regression
            continue
            
        # Convert to numpy arrays and calculate log values
        log_text_lengths = np.log(text_lengths)
        log_vocab_sizes = np.log(vocab_sizes)
        
        # Linear regression to find Herdan coefficient
        # Herdan's law: V = K * N^β where V is vocabulary size, N is text length,
        # K is a constant, and β is the Herdan coefficient
        # In log space: log(V) = β * log(N) + log(K)
        slope, intercept = np.polyfit(log_text_lengths, log_vocab_sizes, 1)
        herdan_coefficient = slope
        
        # Calculate R-squared
        y_pred = slope * log_text_lengths + intercept
        ss_total = np.sum((log_vocab_sizes - np.mean(log_vocab_sizes))**2)
        ss_residual = np.sum((log_vocab_sizes - y_pred)**2)
        if ss_total == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_residual / ss_total)
        
        herdan_coefficients.append(herdan_coefficient)
        r_squared_values.append(r_squared)
    
    # Calculate average values
    if not herdan_coefficients:
        return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
        
    avg_herdan_coefficient = sum(herdan_coefficients) / len(herdan_coefficients)
    avg_r_squared = sum(r_squared_values) / len(r_squared_values)
    
    # Return the average values and some additional information
    return {
        'coefficient': avg_herdan_coefficient,
        'r_squared': avg_r_squared,
        'data': {
            'individual_coefficients': herdan_coefficients,
            'individual_r_squared': r_squared_values
        }
    }

def count_words(text: str) -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text: Text string to analyze
        
    Returns:
        Integer count of words
    """
    return len(re.findall(r'\w+', text))

def get_word_counts(prompts: List[str], reasoning: List[str]) -> Dict[str, int]:
    """
    Calculate word counts for prompts and reasoning.
    
    Args:
        prompts: List of prompt texts
        reasoning: List of reasoning texts
        
    Returns:
        Dictionary with word count statistics
    """
    prompt_word_count = sum(count_words(text) for text in prompts)
    reasoning_word_count = sum(count_words(text) for text in reasoning)
    total_word_count = prompt_word_count + reasoning_word_count
    
    return {
        'prompt_words': prompt_word_count,
        'reasoning_words': reasoning_word_count,
        'total_words': total_word_count
    }