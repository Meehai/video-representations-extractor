# pylint: disable=all

def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
    """Compute the output size given input size and target short edge length."""
    scale = short_edge_length / min(oldh, oldw)
    newh, neww = (short_edge_length, scale * oldw) if oldh < oldw else (scale * oldh, short_edge_length)
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh, neww = newh * scale, neww * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    return newh, neww
