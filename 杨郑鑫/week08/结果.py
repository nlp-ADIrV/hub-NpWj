"""
python evaluate.py --model_type biencoder --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt

              precision    recall  f1-score   support

         不相似       0.89      0.83      0.86      4291
          相似       0.85      0.90      0.87      4329

    accuracy                           0.87      8620
   macro avg       0.87      0.87      0.87      8620
weighted avg       0.87      0.87      0.87      8620

python evaluate.py --model_type biencoder --ckpt ../outputs/checkpoints/biencoder_triplet_best.pt
              precision    recall  f1-score   support

         不相似       0.87      0.83      0.85      4291
          相似       0.84      0.88      0.86      4329

    accuracy                           0.86      8620
   macro avg       0.86      0.86      0.86      8620
weighted avg       0.86      0.86      0.86      8620

python evaluate.py --model_type crossencoder --ckpt ../outputs/checkpoints/crossencoder_best.pt

CrossEncoder 评估结果（validation，8620 条）
  Accuracy: 0.8778
  F1      : 0.8778

              precision    recall  f1-score   support

         不相似       0.88      0.87      0.88      4291
          相似       0.88      0.88      0.88      4329

    accuracy                           0.88      8620
   macro avg       0.88      0.88      0.88      8620
weighted avg       0.88      0.88      0.88      8620

python compare_methods.py
 biencoder_cosine                0.8677        0.8676  threshold=0.66
  biencoder_triplet               0.8556        0.8555  threshold=0.53
  crossencoder                    0.8778        0.8778          argmax
  结论速览：
  最高 Accuracy : crossencoder (0.8778)
  最高 F1       : crossencoder  (0.8778)

  Cosine vs Triplet (Δ):
    Accuracy: -0.0122  F1: -0.0121
    → CosineEmbeddingLoss 更优，AFQMC 数据量下直接对标签优化更稳定
"""
