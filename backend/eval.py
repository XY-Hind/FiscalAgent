import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from backend.services.graph import rag_agent

load_dotenv()

async def run_real_pipeline(question: str):
    """
    Executes the actual RAG pipeline for evaluation.
    """
    config = {"configurable": {"thread_id": "eval-session"}, "recursion_limit": 10}
    input_state = {"messages": [("user", question)]}
    
    # We call the real graph here
    result = await rag_agent.ainvoke(input_state, config=config)
    
    # Extract data for Ragas
    final_output = result.get("final_output")
    context = result.get("retrieved_context", "")
    
    return {
        "answer": final_output.answer if final_output else "No answer generated.",
        "contexts": [context]
    }

async def main():
    print("Starting production-ready evaluation...")
    
    # TODO: Replace placeholder questions with a real curated golden dataset.
    # TODO: Load dataset from cloud storage (e.g. Supabase table or S3 bucket).
    questions = ["What was the total revenue growth?"] 
    ground_truths = ["The revenue growth was 20%."] 
    
    answers = []
    contexts = []
    
    for q in questions:
        res = await run_real_pipeline(q)
        answers.append(res["answer"])
        contexts.append(res["contexts"])
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    # Groq as evaluator for speed/cost
    evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Evaluation
    # try:
    #     result = evaluate(
    #         dataset = dataset, 
    #         metrics=[faithfulness, answer_relevance, context_recall],
    #         llm=evaluator_llm,
    #         embeddings=evaluator_embeddings
    #     )
    #     print(result)
    # except Exception as e:
    #     print(f"Eval Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
