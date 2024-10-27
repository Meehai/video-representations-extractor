"""BaseMixin required to call all constructors in the inheritance chain"""
class BaseMixin:
    """BaseMixin required to call all constructors in the inheritance chain"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
