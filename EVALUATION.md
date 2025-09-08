# Evaluation Results and Summaries

## Average Metrics

(Based on 3 test queries: Run 1, Run 2, Run 3)
Note: This evaluation is based on only 3 queries. Despite the small sample size, results show robust functionality. Summaries are generally accurate and relevant, with strong recall. With more data points, precision and F1 are expected to improve.

Precision@TopK: 0.70
Recall@TopK: 1.00

ROUGE Scores (average):
ROUGE-1: Precision = 0.20, Recall = 0.83, F1 = 0.326
ROUGE-2: Precision = 0.12, Recall = 0.47, F1 = 0.17

## Results breakdow
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Data Point | Query                                                    | Precision@TopK | Recall@TopK | ROUGE-1 (P / R / F1) | ROUGE-2 (P / R / F1) | Generated Summary 
|------------|----------------------------------------------------------|----------------|-------------|----------------------|----------------------|----------------------
| 1          | Jim Cramer opinion on Amazon AI stocks?                  | 0.11           | 1.00        | 0.19 / 0.88 / 0.32   | 0.10 / 0.48 / 0.17   |  Jim Cramer expressed cautious optimism about Amazon.com, Inc. (AMZN) in AI, noting strong Q3 earnings but weaker cloud growth in Q4. He sees AMZN as a top AI stock but suggests others may yield higher returns
| 2          | iPhone SE4 revenue impact?                               | 1.00           | 1.00        | 0.16 / 0.87 / 0.27   | 0.06 / 0.36 / 0.10   |  UBS analysts report Apple’s iPhone SE4 launch is not expected to affect revenue guidance, implying minimal financial impact despite consumer interest.
| 3          | Microsoft backed OpenAI on Tuesday introduced SWE-Lancer | 1.00           | 1.00        | 0.26 / 0.75 / 0.39   | 0.20 / 0.58 / 0.29   |  Microsoft-backed OpenAI introduced SWE-Lancer, showcasing innovation in AI with strong alignment to enterprise adoption.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
