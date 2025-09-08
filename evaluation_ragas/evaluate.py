import asyncio
from pipeline.graph import build_rag_graph
from pipeline.nodes import PipelineState
from utils.logging import setup_logger
from rouge_score import rouge_scorer

# -------------------------
# Logger
# -------------------------
logger = setup_logger("eval")

# -------------------------
# Test Dataset
# -------------------------
# Each item contains:
#   - query: user question
#   - relevant_titles: list of titles that should be retrieved
#   - reference_summary: ground-truth summary for evaluation
test_data = [
   {
        "query": " Microsoft backed OpenAI on Tuesday introduced SWE-Lancer",
        "relevant_titles": [
            "Microsoft-Backed OpenAI Introduces SWE-Lancer Benchmark"
        ],
        "reference_summary": "Microsoft-backed OpenAI introduced SWE-Lancer, a benchmark consisting of over 1,400 free resources for evaluating software engineering performance."
    },
    {
        "query": "iPhone SE4 revenue impact?",
        "relevant_titles": [
            "Apple's Launch of iPhone SE4 Not Seen Impacting Revenue Guidance, UBS Says"
        ],
        "reference_summary": "The iPhone SE4 launch is unlikely to affect Apple's revenue guidance, according to UBS."
    },
     {
        "query": "Jim Cramer opinion on Amazon AI stocks?",
        "relevant_titles": [
            "Jim Cramer on Amazon.com (AMZN): ‘Knock Yourself Out And Sell It If You Have To’"
        ],
        "reference_summary": "Jim Cramer discussed Amazon.com (AMZN) as a key player in eCommerce and cloud computing with AI exposure. He noted factors like energy spending and GPU orders influencing AI stock performance. Since his remarks, AMZN shares gained 26.5% due to strong earnings and optimism about AI-driven revenue."
    }
]

# -------------------------
# Evaluation Functions
# -------------------------
def evaluate_retrieval(retrieved_docs, relevant_titles):
     """
    Compute Precision and Recall for retrieved documents against relevant titles.

    Args:
        retrieved_docs (list): List of document objects from RAG pipeline.
        relevant_titles (list): Ground-truth list of relevant titles.

    Returns:
        tuple: (precision, recall, list of retrieved titles)
    """
    retrieved_titles = [doc.payload.get("title", "") for doc in retrieved_docs]
    num_relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant_titles)
    precision = num_relevant_retrieved / len(retrieved_titles) if retrieved_titles else 0
    recall = num_relevant_retrieved / len(relevant_titles) if relevant_titles else 0
    return precision, recall, retrieved_titles

def evaluate_summary(pred_summary, ref_summary):
    """
    Evaluate generated summary using ROUGE metrics.

    Args:
        pred_summary (str): Generated summary from RAG pipeline.
        ref_summary (str): Reference (ground-truth) summary.

    Returns:
        dict: ROUGE scores for rouge1, rouge2, and rougeL
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref_summary, pred_summary)
    return scores

# -------------------------
# Run RAG Graph Programmatically
# -------------------------
async def run_rag_query(query, chat_history=""):
     """
    Execute RAG pipeline programmatically for a single query.

    Args:
        query (str): User query string
        chat_history (str, optional): Previous chat context

    Returns:
        PipelineState: Final pipeline state with articles and summary
    """
    initial_state = {
        "query": query,
        "chat_history": chat_history,
        "articles": [],
        "summary": "",
        "ticker": ""
    }
    graph = build_rag_graph()
    final_state = await graph.ainvoke(initial_state)
    return final_state

# -------------------------
# Main Evaluation Loop
# -------------------------
async def evaluate_test_set(test_data):
     """
    Loop over test dataset, run RAG, evaluate retrieval and summary, and print metrics.

    Args:
        test_data (list): List of dicts containing query, relevant_titles, reference_summary
    """
    for item in test_data:
        query = item["query"]
        relevant_titles = item["relevant_titles"]
        reference_summary = item["reference_summary"]

        final_state = await run_rag_query(query)
        retrieved_docs = final_state.get("articles", [])
        summary = final_state.get("summary", "")

        # Evaluate retrieval
        precision, recall, retrieved_titles = evaluate_retrieval(retrieved_docs, relevant_titles)
        
        # Evaluate summary
        rouge_scores = evaluate_summary(summary, reference_summary)

        # Print results
        print("="*80)
        print(f"Query: {query}")
        print(f"Relevant Titles: {relevant_titles}")
        print(f"Retrieved Titles: {retrieved_titles}")
        print(f"Precision@TopK: {precision:.2f}, Recall@TopK: {recall:.2f}")
        print("ROUGE Scores:")
        for k, v in rouge_scores.items():
            print(f"  {k}: Precision={v.precision:.2f}, Recall={v.recall:.2f}, F1={v.fmeasure:.2f}")
        print("Generated Summary:\n", summary)
        print("="*80 + "\n")

# -------------------------
# Run Evaluation
# -------------------------
if __name__ == "__main__":
    asyncio.run(evaluate_test_set(test_data))
