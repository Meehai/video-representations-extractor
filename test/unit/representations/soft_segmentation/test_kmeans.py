import numpy as np
from vre.representations.soft_segmentation.kmeans import KMeans
from vre.utils import FakeVideo

def test_kmeans():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    kmeans_repr = KMeans(name="dpt", dependencies=[], n_clusters=np.random.randint(2, 10), epsilon=0.1,
                         max_iterations=2, attempts=1)

    frames = np.array(video[0:1])
    data, extra = (y_kmeans := kmeans_repr(frames))
    assert data.shape == (1, 64, 128, kmeans_repr.n_clusters), data.shape
    assert len(extra) == 1, extra
    assert extra[0]["centers"].shape == (kmeans_repr.n_clusters, 3)
    data_images = kmeans_repr.make_images(frames, y_kmeans)
    assert data_images.shape == (1, 64, 128, 3), data_images.shape
    assert data_images.dtype == np.uint8, data_images.dtype

    assert kmeans_repr.size(y_kmeans) == (64, 128)
    y_kmeans_resized = kmeans_repr.resize(y_kmeans, (32, 64))
    assert kmeans_repr.size(y_kmeans_resized) == (32, 64)
    assert kmeans_repr.make_images(frames, y_kmeans_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_kmeans()
