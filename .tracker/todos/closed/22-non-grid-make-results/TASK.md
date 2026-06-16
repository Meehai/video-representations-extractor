# support for non map/grid/images result in make()

**Created**: 2023-11-10
**Closed**: 2023-11-10
**Priority**: 3

## Description

- `make` should return any sort of arrays, dicts/list/tuples of arrays. Basically, it should be savable by np.save() (even pickle stuff)
  - `y: (np.ndarray | iterable[np.ndarray], dict) = make(t)`
- `make_images` should get the original image as well as the result of `make`
   - `y_img: np.ndarray[T,H,W,3] = make_images(t, y)`
- `resize` should be abstract as well as we cannot make it automagically for all non-map types
  - we should avoid the temptation to create a `MapRepresentation` because this will lead to `SemanticRepresentation`, `BBoxRepresentation` etc.
  - OOP is good, but this level of inheritance is not --> each representation should implement its thing
- prerequisite for SAM (bbox representations in general)
