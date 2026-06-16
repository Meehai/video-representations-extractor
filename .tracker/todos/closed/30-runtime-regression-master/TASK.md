# runtime regression on master compared to a few commits ago

**Created**: 2023-11-17
**Closed**: 2024-05-05
**Priority**: 3
**Labels**: docs/examples/user api

## Description

<table>
<tr>
<td>

</td>
<td>batch=1</td>
<td>batch=3</td>
<td>ratio 1/3</td>
<td>batch=5</td>
<td>ratio 1/5</td>
</tr>
<tr>
<td>mask2former (swin)</td>
<td align="right">5.288</td>
<td align="right">1.748</td>
<td align="right">3.026</td>
<td align="right">1.074</td>
<td align="right">4.926</td>
</tr>
<tr>
<td>mask2former (r50)</td>
<td align="right">6.127</td>
<td align="right">715827882.7</td>
<td align="right">0</td>
<td align="right">1.263</td>
<td align="right">4.85</td>
</tr>
<tr>
<td>fastsam (x)</td>
<td align="right">1.726</td>
<td align="right">0.535</td>
<td align="right">3.226</td>
<td align="right">0.361</td>
<td align="right">4.783</td>
</tr>
<tr>
<td>fastsam (s)</td>
<td align="right">1.747</td>
<td align="right">0.5</td>
<td align="right">3.492</td>
<td align="right">0.362</td>
<td align="right">4.83</td>
</tr>
<tr>
<td>opticalflow raft</td>
<td align="right">1.953</td>
<td align="right">0.643</td>
<td align="right">3.039</td>
<td align="right">0.421</td>
<td align="right">4.642</td>
</tr>
<tr>
<td>halftone</td>
<td align="right">4.474</td>
<td align="right">1.471</td>
<td align="right">3.042</td>
<td align="right">1.001</td>
<td align="right">4.471</td>
</tr>
<tr>
<td>semantic safeuav torch</td>
<td align="right">0.128</td>
<td align="right">0.04</td>
<td align="right">3.152</td>
<td align="right">0.028</td>
<td align="right">4.639</td>
</tr>
<tr>
<td>opticalflow rife</td>
<td align="right">4.251</td>
<td align="right">1.411</td>
<td align="right">3.012</td>
<td align="right">0.96</td>
<td align="right">4.426</td>
</tr>
<tr>
<td>depth dpt</td>
<td align="right">0.314</td>
<td align="right">0.106</td>
<td align="right">2.956</td>
<td align="right">0.07</td>
<td align="right">4.46</td>
</tr>
<tr>
<td>canny</td>
<td align="right">0.32</td>
<td align="right">0.107</td>
<td align="right">2.992</td>
<td align="right">0.071</td>
<td align="right">4.497</td>
</tr>
<tr>
<td>softseg kmeans</td>
<td align="right">1.818</td>
<td align="right">0.598</td>
<td align="right">3.041</td>
<td align="right">0.409</td>
<td align="right">4.451</td>
</tr>
<tr>
<td>softseg gb</td>
<td align="right">2.888</td>
<td align="right">0.961</td>
<td align="right">3.004</td>
<td align="right">0.645</td>
<td align="right">4.475</td>
</tr>
<tr>
<td>dexined</td>
<td align="right">1.613</td>
<td align="right">0.534</td>
<td align="right">3.023</td>
<td align="right">0.326</td>
<td align="right">4.947</td>
</tr>
<tr>
<td>hsv</td>
<td align="right">1.264</td>
<td align="right">0.423</td>
<td align="right">2.989</td>
<td align="right">0.254</td>
<td align="right">4.977</td>
</tr>
<tr>
<td>rgb</td>
<td align="right">0.912</td>
<td align="right">0.304</td>
<td align="right">3</td>
<td align="right">0.182</td>
<td align="right">4.997</td>
</tr>
<tr>
<td>depth odoflow (raft)</td>
<td align="right">0.968</td>
<td align="right">0.285</td>
<td align="right">3.401</td>
<td align="right">0.166</td>
<td align="right">5.841</td>
</tr>
<tr>
<td>normals svd (dpth)</td>
<td align="right">7.575</td>
<td align="right">2.503</td>
<td align="right">3.027</td>
<td align="right">1.487</td>
<td align="right">5.093</td>
</tr>
</table>


On d6e944a58bd51f81882b5a68a49b3f6df0ca3134 (2023.11.16)

|  | batch=1 | batch=3 | ratio 1/3 | batch=5 | ratio 1/5 |
|--|--------:|--------:|----------:|--------:|----------:|
| fastsam (x) | 1.21695 | 1.08309 | 1.12359 | 1.08043 | 1.12635 |
| fastsam (s) | 1.20297 | 1.08391 | 1.10984 | 1.23997 | 0.970158 |
| opticalflow raft | 1.66827 | 1.63425 | 1.02081 | 1.62326 | 1.02773 |
| halftone | 4.04626 | 4.05261 | 0.998434 | 4.02986 | 1.00407 |
| semantic safeuav torch | 0.116757 | 0.112078 | 1.04176 | 0.110953 | 1.05232 |
| opticalflow rife | 3.29536 | 3.2869 | 1.00257 | 3.287 | 1.00254 |
| depth dpt | 0.283716 | 0.287435 | 0.987062 | 0.285369 | 0.994208 |
| canny | 0.254209 | 0.254058 | 1.00059 | 0.25424 | 0.999878 |
| softseg kmeans | 1.56253 | 1.56018 | 1.00151 | 1.57089 | 0.994678 |
| softseg gb | 2.01462 | 2.00241 | 1.0061 | 1.9945 | 1.01009 |
| dexined | 1.31869 | 1.3153 | 1.00258 | 1.31 | 1.00664 |
| hsv | 1.04115 | 1.04313 | 0.998103 | 1.04468 | 0.996624 |
| rgb | 0.74044 | 0.742626 | 0.997057 | 0.741537 | 0.998521 |
| depth odoflow (raft) | 0.867008 | 0.770709 | 1.12495 | 0.722451 | 1.20009 |
| normals svd (dpth) | 6.17803 | 6.17691 | 1.00018 | 6.42862 | 0.96102 |
| total | 5161.39 | 5081.12 | 1.0158 | 5144.75 | 1.00323 |

On 74d409e204f5c6e0f8229472a58fadfe660b2155 (2023.11.11)

|  | batch=1 | batch=3 | ratio 1/3 | batch=5 | ratio 1/5 |
|--|--------:|--------:|----------:|--------:|----------:|
| rgb | 0.00287311 | 0.00285926 | 1.00485 | 0.00355646 | 0.807857 |
| hsv | 0.227086 | 0.23222 | 0.977893 | 0.234075 | 0.970141 |
| normals svd (dpth) | 3.05688 | 3.04678 | 1.00331 | 3.096 | 0.987362 |
| halftone | 3.1492 | 3.14151 | 1.00245 | 3.13804 | 1.00356 |
| softseg kmeans | 0.821758 | 0.818581 | 1.00388 | 0.816683 | 1.00621 |
| canny | 0.0149038 | 0.0143591 | 1.03793 | 0.0142797 | 1.0437 |
| softseg gb | 0.256591 | 0.244486 | 1.04952 | 0.236025 | 1.08714 |
| dexined | 0.121858 | 0.10822 | 1.12602 | 0.105016 | 1.16038 |
| depth dpt | 0.134931 | 0.121348 | 1.11193 | 0.113324 | 1.19066 |
| depth odoflow (raft) | 1.39505 | 1.17315 | 1.18915 | 1.15518 | 1.20764 |
| fastsam (x) | 0.0565283 | 0.0414641 | 1.36331 | 0.0392864 | 1.43888 |
| opticalflow rife | 0.0435189 | 0.0308007 | 1.41292 | 0.0288509 | 1.50841 |
| opticalflow raft | 0.999548 | 0.742625 | 1.34597 | 0.650921 | 1.53559 |
| fastsam (s) | 0.0348993 | 0.0215417 | 1.62008 | 0.019354 | 1.80321 |
| semantic safeuav torch | 0.0251322 | 0.012268 | 2.04859 | 0.00914755 | 2.74742 |

All are a bit worse (much worse for some...) ![image](/uploads/40c77a73e42f8f800168c18a328fb8b5/image.png) We can start with RGB. Odoflow seems to be the worst (3s \> 6s now)...
