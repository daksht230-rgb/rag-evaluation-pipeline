from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from rag_pipeline import ProductionRAGEvaluator

app = FastAPI(
    title="Production-Grade RAG Evaluation Pipeline",
    description="Automated RAG benchmarking API evaluating context relevance and faithfulness.",
    version="1.0.0"
)

evaluator = ProductionRAGEvaluator()

class RAGTestCase(BaseModel):
    query: str = Field(..., example="What is the capital of France?")
    contexts: List[str] = Field(..., example=["France is a country in Western Europe. Its capital is Paris."])
    response: str = Field(..., example="The capital of France is Paris.")

class RAGBatchRequest(BaseModel):
    eval_cases: List[RAGTestCase]

@app.post("/v1/rag/evaluate", summary="Run Production RAG Evaluation Suite")
def evaluate_rag_pipeline(payload: RAGBatchRequest):
    try:
        formatted_batch = [
            {"query": case.query, "contexts": case.contexts, "response": case.response}
            for case in payload.eval_cases
        ]
        
        df_results = evaluator.run_rag_evaluation(formatted_batch)
        
        passed_count = int((df_results["pipeline_status"] == "PASSED").sum())
        total_count = len(formatted_batch)
        
        return {
            "status": "success",
            "summary": {
                "total_evaluated": total_count,
                "passed_evaluations": passed_count,
                "success_rate_pct": round((passed_count / total_count) * 100, 2) if total_count > 0 else 0.0
            },
            "detailed_metrics": df_results.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
