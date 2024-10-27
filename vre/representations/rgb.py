"""RGB representation. Simply inherit FakeRepresentation that does all this needs (i.e. copy pasta the frames)"""
from .fake_representation import FakeRepresentation

class RGB(FakeRepresentation):
    """RGB representation. Sets output_dtype to uint8"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dtype = "uint8"
