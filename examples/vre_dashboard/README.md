# VRE Dir Analysis

Steps

1. We need to generate the json file via:

```
./vre_dir_analysis root_dir > data.json
```

You can also use a `while true; do... sleep 1; done` construct to fetch.

2. Then, we need to run a small python server (due to CORS stuff):
```
python -m http.server . -p 8901
```

3. And then run open: http://localhost:8901/dashboard.html.

