"""
Comprehensive Metric-Learning Retrieval Evaluation Metrics.

Implements:
1. HitRate@1 (Recall@1) - At least one relevant item ranked first.
2. HitRate@5 (Recall@5) - At least one relevant item in top 5.
3. Recall@10 - At least one relevant item in top 10.
4. Precision@5 - Proportion of top-5 retrieved items that are relevant.
5. mAP@5 - Mean Average Precision at cutoff 5.
6. Leave-one-out query-gallery matching protocol.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_comprehensive_metrics(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    retrieved_indices: np.ndarray,
) -> Dict[str, float]:
    """
    Args:
        query_labels: [N_q] true identity label per query.
        gallery_labels: [N_g] true identity label per gallery candidate.
        retrieved_indices: [N_q, max_k] gallery indices retrieved per query (excluding self).
    Returns:
        Dict with HitRate@1, HitRate@5, Recall@5, Recall@10, Precision@5, mAP@5.
    """
    n_queries = len(query_labels)
    hit_1 = 0.0
    hit_5 = 0.0
    hit_10 = 0.0
    recall_5_list = []
    recall_10_list = []
    precision_5_list = []
    ap_5_list = []
    valid_queries = 0

    for i in range(n_queries):
        true_label = query_labels[i]
        retrieved_ids = gallery_labels[retrieved_indices[i]]

        # Total relevant items in candidate gallery for this query identity
        # When evaluating leave-one-out (query excluded from retrieval), exclude the query from relevant count
        total_matches_in_gallery = np.sum(gallery_labels == true_label)
        if total_matches_in_gallery <= 1:
            # Query has no peer images in the gallery
            continue
        total_relevant_candidates = total_matches_in_gallery - 1
        valid_queries += 1

        # 1. HitRate@1: Is at least one relevant item ranked first?
        if retrieved_ids[0] == true_label:
            hit_1 += 1.0

        # 2. HitRate@5: Is at least one relevant item in top 5?
        if np.any(retrieved_ids[:5] == true_label):
            hit_5 += 1.0

        # 3. HitRate@10: Is at least one relevant item in top 10?
        if np.any(retrieved_ids[:10] == true_label):
            hit_10 += 1.0

        # 4. Precision@5: Fraction of the top 5 results that are relevant (out of 5)
        n_rel_5 = np.sum(retrieved_ids[:5] == true_label)
        precision_5_list.append(n_rel_5 / 5.0)

        # 5. Recall@5: Fraction of all relevant gallery items retrieved in top 5
        recall_5_list.append(n_rel_5 / total_relevant_candidates)

        # 6. Recall@10: Fraction of all relevant gallery items retrieved in top 10
        n_rel_10 = np.sum(retrieved_ids[:10] == true_label)
        recall_10_list.append(n_rel_10 / total_relevant_candidates)

        # 7. Average Precision @ 5 (AP@5)
        # AP@5 = (1 / min(total_relevant_candidates, 5)) * sum_{k=1..5} (P@k * rel(k))
        num_relevant_hits = 0
        prec_sum = 0.0
        for k in range(1, 6):
            if retrieved_ids[k - 1] == true_label:
                num_relevant_hits += 1
                prec_at_k = num_relevant_hits / k
                prec_sum += prec_at_k

        denom = min(total_relevant_candidates, 5)
        ap_5 = (prec_sum / denom) if denom > 0 else 0.0
        ap_5_list.append(ap_5)

    if valid_queries == 0:
        return {
            "HitRate@1": 0.0,
            "HitRate@5": 0.0,
            "HitRate@10": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "Precision@5": 0.0,
            "mAP@5": 0.0,
        }

    return {
        "HitRate@1": round((hit_1 / valid_queries) * 100.0, 2),
        "HitRate@5": round((hit_5 / valid_queries) * 100.0, 2),
        "HitRate@10": round((hit_10 / valid_queries) * 100.0, 2),
        "Recall@5": round(float(np.mean(recall_5_list)) * 100.0, 2),
        "Recall@10": round(float(np.mean(recall_10_list)) * 100.0, 2),
        "Precision@5": round(float(np.mean(precision_5_list)) * 100.0, 2),
        "mAP@5": round(float(np.mean(ap_5_list)), 4),
    }
