import os
import json
from pathlib import Path
from bert_score import score
from rouge_score import rouge_scorer

def compute_scores(generated_dir, data_dir):
    """Compute BERTScore and ROUGE scores between generated patents and input papers."""
    results = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for file in Path(generated_dir).glob("*_patent.md"):
        doc_id = file.stem.replace("_patent", "")
        paper_path = Path(data_dir) / doc_id / "paper.md"
        
        if not paper_path.exists():
            continue
            
        with open(file) as f:
            generated = f.read()
        with open(paper_path) as f:
            reference = f.read()
        
        # BERTScore
        P, R, F1 = score([generated], [reference], lang="en", verbose=False)
        
        # ROUGE scores
        rouge_scores = scorer.score(reference, generated)
        
        results.append({
            "doc_id": doc_id,
            "bertscore": {
                "precision": P.item(),
                "recall": R.item(),
                "f1": F1.item()
            },
            "rouge1": {
                "precision": rouge_scores['rouge1'].precision,
                "recall": rouge_scores['rouge1'].recall,
                "f1": rouge_scores['rouge1'].fmeasure
            },
            "rouge2": {
                "precision": rouge_scores['rouge2'].precision,
                "recall": rouge_scores['rouge2'].recall,
                "f1": rouge_scores['rouge2'].fmeasure
            },
            "rougeL": {
                "precision": rouge_scores['rougeL'].precision,
                "recall": rouge_scores['rougeL'].recall,
                "f1": rouge_scores['rougeL'].fmeasure
            }
        })
    
    return results

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m flowplan_g2p.evaluation.bertscore_eval <generated_dir> <data_dir>")
        sys.exit(1)

    generated_dir = sys.argv[1]
    data_dir = sys.argv[2]

    results = compute_scores(generated_dir, data_dir)
    if not results:
        print("No results found.")
        sys.exit(1)

    def _avg(metric: str, key: str) -> float:
        return sum(r[metric][key] for r in results) / len(results)

    for metric in ("bertscore", "rouge1", "rouge2", "rougeL"):
        print(f"{metric.upper()} - P: {_avg(metric, 'precision'):.4f}, "
              f"R: {_avg(metric, 'recall'):.4f}, F1: {_avg(metric, 'f1'):.4f}")

    with open("evaluation_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
