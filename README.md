# catboost_inference

Repo to try high-performance inference using catboost.

Archive:
Example showing issue https://github.com/catboost/catboost/issues/1835

Thes scripts simulates/checks a memory leak in 
an environment where a online prediction service
will receive requests and return predictions continuously
for small batches.


```
pip install catboost psutil


# To test memory leak on creating a pool and calling model.predict
python pool_leak.py

# To test memory leak on calling model.predict directly with a list
python predict_leak.py

# To test memory leak only on creating a Pool (no model loaded)
python poolonly_leak.py
```

In order to see the leak, you need to run a high number of iterations > 1 million with a small batch size (i.e 150)
