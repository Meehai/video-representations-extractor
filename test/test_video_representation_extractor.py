import gdown
from video_representations_extractor import VideoRepresentationsExtractor
from pathlib import Path

class TestVideoRepresentationExtractor:
    @staticmethod
    def setup():
        if not Path("testVideo.mp4").exists():
            gdown.download("https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk", "testVideo.mp4")

    def test_video_representation_extractor_1(self):
        TestVideoRepresentationExtractor.setup()
        vre = VideoRepresentationsExtractor("testVideo.mp4", "tmp/vre/", \
            {"rgb": {"type":"default", "method":"rgb", "dependencies":[], "parameters":{}}})
        assert not vre is None
        vre.doExport(1000, 1001, exportCollage=True)
        assert Path("tmp/vre/rgb/1000.npz").exists()
        assert Path("tmp/vre/collage/1000.png").exists()
