"""Load customer profiles with one row per customer."""
from ..datasets.storage import DatasetError


def customer_frame(table):
    if table is None or not table.num_rows:
        raise DatasetError("Dataset has no customer profiles table", "customer_profiles_missing", 422)
    customers = table.to_pandas().sort_values("customer_id", ignore_index=True)
    if customers["customer_id"].duplicated().any():
        raise DatasetError("Expected one profile per customer_id", "duplicate_customer_id", 422)
    return customers
