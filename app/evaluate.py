#!/usr/bin/env python3
"""
Evaluation Script for Legal Assistant Chatbot
Measures accuracy, response quality, and performance metrics
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gemini_service import GeminiLegalService
from src.rag_model import RAGLegalAssistant
from src.database import LegalDatabase
from src.config import Config

# Try to import evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  nltk not installed. Install with: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotEvaluator:
    """Evaluates chatbot performance and accuracy"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.db = LegalDatabase()
        self.gemini_service = GeminiLegalService(db=self.db)
        
        # Try to initialize RAG model
        try:
            self.rag_model = RAGLegalAssistant()
            self.gemini_service.rag_model = self.rag_model
            self.has_rag = True
        except Exception as e:
            logger.warning(f"RAG model not available: {str(e)}")
            self.has_rag = False
        
        # Initialize metrics calculators
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if SEMANTIC_AVAILABLE:
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def evaluate_faq_accuracy(self, test_questions: List[Dict]) -> Dict:
        """
        Evaluate FAQ matching accuracy
        
        Args:
            test_questions: List of dicts with 'question' and 'expected_faq_id' or 'expected_answer'
        
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Evaluating FAQ matching accuracy...")
        
        correct = 0
        total = len(test_questions)
        results = []
        
        for i, test_case in enumerate(test_questions, 1):
            question = test_case['question']
            expected_faq_id = test_case.get('expected_faq_id')
            expected_answer = test_case.get('expected_answer', '')
            
            # Get response from chatbot
            response = self.gemini_service.get_legal_guidance(question)
            
            if response.get('success'):
                # Check if FAQ was matched
                matched_faq_id = response.get('faq_id')
                matched_answer = response.get('answer', '')
                
                # Evaluate match
                is_correct = False
                if expected_faq_id and matched_faq_id:
                    is_correct = (expected_faq_id == matched_faq_id)
                elif expected_answer:
                    # Use semantic similarity if available
                    if SEMANTIC_AVAILABLE:
                        similarity = self._semantic_similarity(expected_answer, matched_answer)
                        is_correct = similarity > 0.7
                    else:
                        # Simple keyword matching
                        expected_keywords = set(expected_answer.lower().split())
                        matched_keywords = set(matched_answer.lower().split())
                        overlap = len(expected_keywords & matched_keywords) / len(expected_keywords) if expected_keywords else 0
                        is_correct = overlap > 0.5
                
                if is_correct:
                    correct += 1
                
                results.append({
                    'question': question,
                    'expected_faq_id': expected_faq_id,
                    'matched_faq_id': matched_faq_id,
                    'correct': is_correct
                })
            else:
                results.append({
                    'question': question,
                    'error': response.get('error', 'Unknown error'),
                    'correct': False
                })
            
            if i % 10 == 0:
                logger.info(f"Processed {i}/{total} questions...")
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }
    
    def evaluate_response_quality(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate response quality using ROUGE, BLEU, and semantic similarity
        
        Args:
            test_cases: List of dicts with 'question', 'expected_answer', and optionally 'reference_answer'
        
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Evaluating response quality...")
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        semantic_scores = []
        response_times = []
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case['question']
            expected_answer = test_case.get('expected_answer', '')
            reference_answer = test_case.get('reference_answer', expected_answer)
            
            # Measure response time
            start_time = time.time()
            response = self.gemini_service.get_legal_guidance(question)
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            if response.get('success'):
                generated_answer = response.get('answer', '')
                
                # Calculate ROUGE scores
                if ROUGE_AVAILABLE and reference_answer:
                    rouge_scores_dict = self.rouge_scorer.score(reference_answer, generated_answer)
                    for metric in rouge_scores:
                        rouge_scores[metric].append(rouge_scores_dict[metric].fmeasure)
                
                # Calculate BLEU score
                if BLEU_AVAILABLE and reference_answer:
                    try:
                        reference_tokens = word_tokenize(reference_answer.lower())
                        generated_tokens = word_tokenize(generated_answer.lower())
                        bleu = sentence_bleu([reference_tokens], generated_tokens, 
                                            smoothing_function=SmoothingFunction().method1)
                        bleu_scores.append(bleu)
                    except Exception as e:
                        logger.warning(f"BLEU calculation failed: {str(e)}")
                
                # Calculate semantic similarity
                if SEMANTIC_AVAILABLE and reference_answer:
                    similarity = self._semantic_similarity(reference_answer, generated_answer)
                    semantic_scores.append(similarity)
            
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(test_cases)} test cases...")
        
        # Calculate averages
        metrics = {
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
        }
        
        if rouge_scores['rougeL']:
            metrics['rouge1'] = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
            metrics['rouge2'] = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
            metrics['rougeL'] = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
        
        if bleu_scores:
            metrics['bleu'] = sum(bleu_scores) / len(bleu_scores)
        
        if semantic_scores:
            metrics['semantic_similarity'] = sum(semantic_scores) / len(semantic_scores)
        
        return metrics
    
    def evaluate_retrieval_accuracy(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate RAG retrieval accuracy (precision, recall, F1)
        
        Args:
            test_queries: List of dicts with 'query' and 'relevant_doc_ids'
        
        Returns:
            Dictionary with retrieval metrics
        """
        if not self.has_rag:
            return {'error': 'RAG model not available'}
        
        logger.info("Evaluating retrieval accuracy...")
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case['query']
            relevant_doc_ids = set(test_case.get('relevant_doc_ids', []))
            
            # Retrieve documents
            retrieved_docs = self.rag_model.retrieve_relevant_documents(query)
            retrieved_doc_ids = {doc.get('id') for doc in retrieved_docs if doc.get('id')}
            
            # Calculate precision, recall, F1
            if retrieved_doc_ids:
                true_positives = len(relevant_doc_ids & retrieved_doc_ids)
                precision = true_positives / len(retrieved_doc_ids) if retrieved_doc_ids else 0
                recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(test_queries)} queries...")
        
        return {
            'precision': sum(precisions) / len(precisions) if precisions else 0,
            'recall': sum(recalls) / len(recalls) if recalls else 0,
            'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            'total_queries': len(test_queries)
        }
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not SEMANTIC_AVAILABLE:
            return 0.0
        
        try:
            embeddings = self.semantic_model.encode([text1, text2])
            # Cosine similarity
            import numpy as np
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
            return 0.0
    
    def evaluate_comprehensive(self, test_dataset: Dict) -> Dict:
        """
        Run comprehensive evaluation
        
        Args:
            test_dataset: Dictionary with 'faq_tests', 'quality_tests', 'retrieval_tests'
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info("="*60)
        logger.info("Starting Comprehensive Evaluation")
        logger.info("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'has_rag': self.has_rag
        }
        
        # FAQ Accuracy
        if 'faq_tests' in test_dataset:
            logger.info("\n1. Evaluating FAQ Accuracy...")
            faq_results = self.evaluate_faq_accuracy(test_dataset['faq_tests'])
            results['faq_accuracy'] = faq_results
            logger.info(f"   FAQ Accuracy: {faq_results['accuracy']:.2f}%")
        
        # Response Quality
        if 'quality_tests' in test_dataset:
            logger.info("\n2. Evaluating Response Quality...")
            quality_results = self.evaluate_response_quality(test_dataset['quality_tests'])
            results['response_quality'] = quality_results
            if 'rougeL' in quality_results:
                logger.info(f"   ROUGE-L: {quality_results['rougeL']:.4f}")
            if 'semantic_similarity' in quality_results:
                logger.info(f"   Semantic Similarity: {quality_results['semantic_similarity']:.4f}")
            logger.info(f"   Avg Response Time: {quality_results['avg_response_time']:.2f}s")
        
        # Retrieval Accuracy
        if 'retrieval_tests' in test_dataset and self.has_rag:
            logger.info("\n3. Evaluating Retrieval Accuracy...")
            retrieval_results = self.evaluate_retrieval_accuracy(test_dataset['retrieval_tests'])
            results['retrieval_accuracy'] = retrieval_results
            if 'f1_score' in retrieval_results:
                logger.info(f"   F1-Score: {retrieval_results['f1_score']:.4f}")
                logger.info(f"   Precision: {retrieval_results['precision']:.4f}")
                logger.info(f"   Recall: {retrieval_results['recall']:.4f}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Evaluation Summary")
        logger.info("="*60)
        
        # Check against target metrics
        targets_met = {}
        if 'response_quality' in results:
            q = results['response_quality']
            targets_met['rougeL_target'] = q.get('rougeL', 0) >= 0.75
            targets_met['response_time_target'] = q.get('avg_response_time', 999) < 2.0
        
        if 'retrieval_accuracy' in results:
            r = results['retrieval_accuracy']
            targets_met['f1_target'] = r.get('f1_score', 0) >= 0.85
        
        results['targets_met'] = targets_met
        
        return results


def load_test_dataset(file_path: str = None) -> Dict:
    """Load test dataset from JSON file"""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_dataset.json')
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.warning(f"Test dataset not found at {file_path}. Using sample dataset.")
        return get_sample_dataset()


def get_sample_dataset() -> Dict:
    """Get sample test dataset"""
    return {
        'faq_tests': [
            {
                'question': 'Mere shohar ne talaq ka notice bheja hai, main kya karun?',
                'expected_faq_id': None,
                'expected_answer': 'talaq'
            },
            {
                'question': 'Police mein FIR darj karane ka tarika kya hai?',
                'expected_faq_id': None,
                'expected_answer': 'FIR'
            }
        ],
        'quality_tests': [
            {
                'question': 'Tenant ko kese nikalein?',
                'expected_answer': 'Tenant ko nikalne ke liye proper legal notice dena zaroori hai',
                'reference_answer': 'Tenant ko nikalne ke liye proper legal notice dena zaroori hai'
            }
        ],
        'retrieval_tests': []
    }


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Legal Assistant Chatbot')
    parser.add_argument('--dataset', type=str, help='Path to test dataset JSON file')
    parser.add_argument('--output', type=str, help='Path to save evaluation results JSON')
    parser.add_argument('--faq-only', action='store_true', help='Only evaluate FAQ accuracy')
    parser.add_argument('--quality-only', action='store_true', help='Only evaluate response quality')
    parser.add_argument('--retrieval-only', action='store_true', help='Only evaluate retrieval')
    
    args = parser.parse_args()
    
    # Load test dataset
    test_dataset = load_test_dataset(args.dataset)
    
    # Initialize evaluator
    evaluator = ChatbotEvaluator()
    
    # Run evaluation
    if args.faq_only:
        results = {'faq_accuracy': evaluator.evaluate_faq_accuracy(test_dataset.get('faq_tests', []))}
    elif args.quality_only:
        results = {'response_quality': evaluator.evaluate_response_quality(test_dataset.get('quality_tests', []))}
    elif args.retrieval_only:
        results = {'retrieval_accuracy': evaluator.evaluate_retrieval_accuracy(test_dataset.get('retrieval_tests', []))}
    else:
        results = evaluator.evaluate_comprehensive(test_dataset)
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {args.output}")
    else:
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

