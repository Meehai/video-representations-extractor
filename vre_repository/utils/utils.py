import numpy as np

def semantic_mapper(semantic_original: np.ndarray, mapping: dict[str, list[str]],
                    original_classes: list[str]) -> np.ndarray:
    """maps a bigger semantic segmentation to a smaller one"""
    assert len(semantic_original.shape) == 2, f"Only argmaxed data supported, got: {semantic_original.shape}"
    assert np.issubdtype(semantic_original.dtype, np.integer), semantic_original.dtype
    mapping_ix = {list(mapping.keys()).index(k): [original_classes.index(_v) for _v in v] for k, v in mapping.items()}
    flat_mapping = {}
    for k, v in mapping_ix.items():
        for _v in v:
            flat_mapping[_v] = k
    assert (A := len(set(flat_mapping))) == (B := len(set(original_classes))), (A, B)
    mapped_data = np.vectorize(flat_mapping.get)(semantic_original).astype(np.uint8)
    return mapped_data
