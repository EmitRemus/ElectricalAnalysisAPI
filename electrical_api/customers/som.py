"""Train Kohonen maps for customer profiles."""
import numpy as np
from minisom import MiniSom

from ..datasets.storage import DatasetError
from .segmentation import prepare_customer_features, train_kmeans, validate_cluster_count


GRID_SIZE = 15
ITERATIONS = 20_000
LEARNING_RATE = 0.5
RANDOM_SEED = 0
RADII = (1.5, 0.5)


def train_som(features, sigma):
    som = MiniSom(GRID_SIZE, GRID_SIZE, features.shape[1], sigma=sigma,
                  learning_rate=LEARNING_RATE, random_seed=RANDOM_SEED)
    som.pca_weights_init(features)
    som.train(features, ITERATIONS, random_order=True)
    return som


def customer_maps(table, k):
    customers, features = prepare_customer_features(table)
    validate_cluster_count(features, k)
    maps = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            labels = train_kmeans(features, k).labels_
            if len(np.unique(labels)) < k:
                raise DatasetError("K-means could not form k distinct segments", "insufficient_segmentation_data", 422)

            for sigma in RADII:
                som = train_som(features, sigma)
                error = float(som.topographic_error(features))
                distance_map = som.distance_map()
                if not np.isfinite(error) or not np.isfinite(distance_map).all():
                    raise FloatingPointError("Nonfinite SOM results")
                winners = (som.winner(vector) for vector in features)
                assignments = [
                    {"customer_id": customer_id, "segment_id": label, "x": int(x), "y": int(y)}
                    for customer_id, label, (x, y) in zip(customers["customer_id"], labels.tolist(), winners)
                ]
                maps.append({"sigma": sigma, "topographic_error": error,
                             "distance_map": distance_map.tolist(), "assignments": assignments})
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise DatasetError("SOM is outside the numeric range", "som_numeric_range", 422) from error

    return {
        "k": k,
        "customer_count": len(customers),
        "grid_size": GRID_SIZE,
        "iterations": ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "maps": maps,
    }
