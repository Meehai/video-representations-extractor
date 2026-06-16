# safeuav sseg batched_test fails in equals

**Created**: 2023-11-10
**Closed**: 2023-11-10
**Priority**: 3

## Description

```
    # we'll just pick 2 random representations to test here
    while True:
        representations_dict = sample_representations(all_representations_dict, n=2)
        if "semantic safeuav torch" in representations_dict.keys():
            break

```

I think it has to do with weights init
