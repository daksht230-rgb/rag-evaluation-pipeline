import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ProductionRAGEvaluator:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model

    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

    def evaluate_context_relevance(self, query: str, retrieved_contexts: List[str]) -> float:
        """Evaluates how relevant the retrieved context is to the user query."""
        if not retrieved_contexts:
            return 0.0
        
        query_tokens = set(query.lower().split())
        relevance_scores = []
        
        for ctx in retrieved_contexts:
            ctx_tokens = set(ctx.lower().split())
            if not query_tokens or not ctx_tokens:
                relevance_scores.append(0.0)
                continue
            intersection = query_tokens.intersection(ctx_tokens)
            score = len(intersection) / len(query_tokens)
            relevance_scores.append(score)
            
        return round(float(np.mean(relevance_scores) * 100.0), 2)

    def evaluate_faithfulness(self, response: str, retrieved_contexts: List[str]) -> str:
        """Checks if the generated response is strictly grounded in retrieved contexts (Hallucination check)."""
        combined_context = " ".join(retrieved_contexts).lower()
        response_tokens = set(response.lower().split())
        
        if not response_tokens:
            return "NOT_GROUNDED"
            
        supported_count = sum(1 for token in response_tokens if token in combined_context)
        faithfulness_ratio = supported_count / len(response_tokens)
        
        if faithfulness_ratio >= 0.75:
            return "HIGHLY_FAITHFUL"
        elif faithfulness_ratio >= 0.40:
            return "MODERATE_FAITHFUL"
        return "HALLUCINATED_RISK"

    def run_rag_evaluation(self, evaluation_batch: List[Dict[str, Any]]) -> pd.DataFrame:
        results = []
        for idx, item in enumerate(evaluation_batch):
            query = item.get("query", "")
            contexts = item.get("contexts", [])
            response = item.get("response", "")
            
            t_start = time.time()
            context_relevance_score = self.evaluate_context_relevance(query, contexts)
            faithfulness_status = self.evaluate_faithfulness(response, contexts)
            latency = round(time.time() - t_start, 4)
            
            passed = context_relevance_score >= 50.0 and faithfulness_status != "HALLUCINATED_RISK"
            
            results.append({
                "eval_id": f"RAG-EVAL-{idx+1:03d}",
                "context_relevance_pct": context_relevance_score,
                "faithfulness_verdict": faithfulness_status,
                "latency_seconds": latency,
                "pipeline_status": "PASSED" if passed else "FLAGGED_REVIEW"
            })
            
        return pd.DataFrame(results)
