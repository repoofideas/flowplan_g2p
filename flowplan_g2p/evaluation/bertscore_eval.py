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
    generated_dir = "/home/kriswpan/repo/flowplan_g2p/output_sonnet4.5"
    data_dir = "/home/kriswpan/repo/flowplan_g2p/Pap2Pat/data"
    
    results = compute_scores(generated_dir, data_dir)
    
    # Calculate averages
    avg_bertscore = {
        "precision": sum(r["bertscore"]["precision"] for r in results) / len(results),
        "recall": sum(r["bertscore"]["recall"] for r in results) / len(results),
        "f1": sum(r["bertscore"]["f1"] for r in results) / len(results)
    }
    
    avg_rouge1 = {
        "precision": sum(r["rouge1"]["precision"] for r in results) / len(results),
        "recall": sum(r["rouge1"]["recall"] for r in results) / len(results),
        "f1": sum(r["rouge1"]["f1"] for r in results) / len(results)
    }
    
    avg_rouge2 = {
        "precision": sum(r["rouge2"]["precision"] for r in results) / len(results),
        "recall": sum(r["rouge2"]["recall"] for r in results) / len(results),
        "f1": sum(r["rouge2"]["f1"] for r in results) / len(results)
    }
    
    avg_rougeL = {
        "precision": sum(r["rougeL"]["precision"] for r in results) / len(results),
        "recall": sum(r["rougeL"]["recall"] for r in results) / len(results),
        "f1": sum(r["rougeL"]["f1"] for r in results) / len(results)
    }
    
    print(f"BERTScore - P: {avg_bertscore['precision']:.4f}, R: {avg_bertscore['recall']:.4f}, F1: {avg_bertscore['f1']:.4f}")
    print(f"ROUGE-1 - P: {avg_rouge1['precision']:.4f}, R: {avg_rouge1['recall']:.4f}, F1: {avg_rouge1['f1']:.4f}")
    print(f"ROUGE-2 - P: {avg_rouge2['precision']:.4f}, R: {avg_rouge2['recall']:.4f}, F1: {avg_rouge2['f1']:.4f}")
    print(f"ROUGE-L - P: {avg_rougeL['precision']:.4f}, R: {avg_rougeL['recall']:.4f}, F1: {avg_rougeL['f1']:.4f}")
    
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "results": results, 
            "averages": {
                "bertscore": avg_bertscore,
                "rouge1": avg_rouge1,
                "rouge2": avg_rouge2,
                "rougeL": avg_rougeL
            }
        }, f, indent=2)
