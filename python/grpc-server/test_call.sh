grpcurl -plaintext -import-path ./proto -proto cb.proto \
    -d '{"cat_feature1": "A", "cat_feature2": "B", "cat_feature3": "C", "float_feature4": 0.5, "float_feature5": 0.33}' \
    localhost:50052 cb.Inference/Predict
