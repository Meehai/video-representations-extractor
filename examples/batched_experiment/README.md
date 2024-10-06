# Batched experiment

Last updated at: 2023.11.11

See [implementation](batched_experiment.py) for cfg and code.

|                        |    batch=1 |    batch=3 |   ratio 1/3 |    batch=5 |   ratio 1/5 |
|:-----------------------|-----------:|-----------:|------------:|-----------:|------------:|
| rgb                    | 0.00287311 | 0.00285926 |    1.00485  | 0.00355646 |    0.807857 |
| hsv                    | 0.227086   | 0.23222    |    0.977893 | 0.234075   |    0.970141 |
| normals svd (dpth)     | 3.05688    | 3.04678    |    1.00331  | 3.096      |    0.987362 |
| halftone               | 3.1492     | 3.14151    |    1.00245  | 3.13804    |    1.00356  |
| canny                  | 0.0149038  | 0.0143591  |    1.03793  | 0.0142797  |    1.0437   |
| softseg gb             | 0.256591   | 0.244486   |    1.04952  | 0.236025   |    1.08714  |
| dexined                | 0.121858   | 0.10822    |    1.12602  | 0.105016   |    1.16038  |
| depth dpt              | 0.134931   | 0.121348   |    1.11193  | 0.113324   |    1.19066  |
| fastsam (x)            | 0.0565283  | 0.0414641  |    1.36331  | 0.0392864  |    1.43888  |
| opticalflow rife       | 0.0435189  | 0.0308007  |    1.41292  | 0.0288509  |    1.50841  |
| opticalflow raft       | 0.999548   | 0.742625   |    1.34597  | 0.650921   |    1.53559  |
| fastsam (s)            | 0.0348993  | 0.0215417  |    1.62008  | 0.019354   |    1.80321  |
| semantic safeuav torch | 0.0251322  | 0.012268   |    2.04859  | 0.00914755 |    2.74742  |

We can easily observe that some representations are not batched yet (all with ratios close to 1). We can also observe
that some of them are super slow (normals_svd, halftone).
