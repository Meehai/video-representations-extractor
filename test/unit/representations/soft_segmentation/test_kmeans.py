import numpy as np
from vre.representations.soft_segmentation.kmeans import KMeans
from vre.utils import FakeVideo

def test_kmeans():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    kmneans_repr = KMeans(name="dpt", dependencies=[], n_clusters=np.random.randint(2, 10), epsilon=0.1,
                          max_iterations=2, attempts=1)

    frames = np.array(video[0:1])
    y_kmeans, extra = kmneans_repr(frames)
    assert y_kmeans.shape == (1, 64, 128, kmneans_repr.n_clusters), y_kmeans.shape
    assert len(extra) == 1, extra
    assert extra[0]["centers"].shape == (kmneans_repr.n_clusters, 3)
    y_kmeans_images = kmneans_repr.make_images(frames, (y_kmeans, extra))
    assert y_kmeans_images.shape == (1, 64, 128, 3), y_kmeans_images.shape
    assert y_kmeans_images.dtype == np.uint8, y_kmeans_images.dtype

if __name__ == "__main__":
    test_kmeans()
