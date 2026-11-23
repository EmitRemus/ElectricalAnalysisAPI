"""Mine association rules for customer churn."""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from ..datasets.storage import DatasetError
from .profiles import customer_frame


DEFAULT_SUPPORTS = (0.05, 0.10)
DEFAULT_CONFIDENCE = 0.30


def churn_rules(basket, min_support, min_confidence):
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    target = frozenset({"churned"})
    if not any(target < itemset for itemset in frequent["itemsets"]):
        return []

    rules = association_rules(frequent, num_itemsets=len(basket), metric="confidence",
                              min_threshold=min_confidence, return_metrics=["support", "confidence", "lift"])
    matching = rules.loc[rules["consequents"] == target]
    results = [
        {"antecedents": sorted(row.antecedents), "consequent": "churned",
         "support": row.support, "confidence": row.confidence, "lift": row.lift}
        for row in matching.itertuples(index=False)
    ]
    return sorted(results, key=lambda rule: (-rule["lift"], rule["antecedents"]))


def report_churn_rules(profiles, event_table, supports, min_confidence):
    customers = customer_frame(profiles)
    if event_table is None or not event_table.num_rows:
        raise DatasetError("Dataset has no customer events table", "customer_events_missing", 422)
    labeled = customers.loc[customers["churned"].notna()]
    if labeled.empty:
        raise DatasetError("Expected at least one known churned value", "churn_labels_missing", 422)

    events = event_table.to_pandas()
    no_churn = events.loc[~events["event_type"].isin(["churn", "churned"])
                          & events["customer_id"].isin(labeled["customer_id"])]
    basket = pd.crosstab(no_churn["customer_id"], no_churn["event_type"]) > 0
    basket = basket.reindex(labeled["customer_id"], fill_value=False)
    basket["churned"] = labeled["churned"].astype(bool).to_numpy()

    churned_count = int(basket["churned"].sum())
    rules = churn_rules(basket, min(supports), min_confidence) if churned_count and basket.shape[1] > 1 else []
    reports = []
    for threshold in supports:
        selected = [rule for rule in rules if rule["support"] >= threshold]
        reports.append({"min_support": threshold, "rule_count": len(selected), "rules": selected})
    return {
        "customer_count": len(customers),
        "labeled_customer_count": len(basket),
        "churned_count": churned_count,
        "churn_rate": churned_count / len(basket),
        "min_confidence": min_confidence,
        "reports": reports,
    }
