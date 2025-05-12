## Сравнение ML-моделей и DL-моделей

Ниже приведены таблицы с метриками для лучших ML-моделей и DL-моделей.

### 1. Лучшие ML-модели

| Классификатор                        | Гиперпараметры                                                                                 | Accuracy | F1 Score | ROC-AUC |
|--------------------------------------|------------------------------------------------------------------------------------------------|----------|----------|---------|
| CatBoost Classifier                  | (default)                                                                                      | 0.599    | 0.609    | 0.647   |
| CatBoost Classifier + PCA            | —                                                                                              | 0.580    | 0.582    | 0.626   |
| CatBoost Classifier + LDA            | —                                                                                              | 0.580    | 0.620    | 0.631   |
| CatBoost Classifier (diff)           | —                                                                                              | 0.592    | 0.605    | 0.649   |
| CatBoost Classifier (tuned)          | iterations=468, depth=6, learning_rate=0.038, l2_leaf_reg=1.76, border_count=207, one_hot_max_size=2 | 0.592    | 0.614    | 0.645   |
| CatBoost Classifier + Autofeat       | —                                                                                              | 0.593    | 0.607    | 0.648   |
| CatBoost Classifier + Heroes         | —                                                                                              | 0.598    | 0.607    | 0.650   |
| Logistic Regression                  | max_iter=500                                                                                   | 0.591    | 0.621    | 0.632   |
| Logistic Regression (mean-features)  | solver='saga', penalty='l1', max_iter=100, C=100                                               | 0.597    | 0.629    | 0.643   |
| Logistic Regression (liblinear, l1)  | solver='liblinear', penalty='l1', max_iter=100, class_weight=None, C=0.01                       | 0.586    | 0.609    | 0.631   |
| Linear Discriminant Analysis (LDA)   | solver='svd', priors=[0.3, 0.7], n_components=1                                                 | 0.527    | 0.680    | 0.633   |

### 2. DL-Модели 

| Модель                        | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------------------|----------|-----------|--------|----------|---------|
| TabNet                        | 0.589    | 0.590     | 0.629  | 0.609    | 0.626   |
| MLP-PyTorch                   | 0.568    | 0.554     | 0.771  | 0.645    | 0.606   |
| MLP-PyTorch + Heroes          | 0.575    | 0.614     | 0.443  | 0.514    | 0.618   |
| TabNet + Heroes               | 0.578    | 0.577     | 0.642  | 0.607    | 0.617   |

### 3. Выводы

- Среди ML-моделей лидером по ROC-AUC выступил **CatBoost Classifier**.
- Модели CatBoost демонстрируют стабильное качество при добавлении PCA или LDA, однако без дополнительных признаков их метрики несколько уступают.
- Линейные модели (Logistic Regression, LDA) показали конкурентные F1-score, но их ROC-AUC обычно ниже моделей CatBoost.
- В сегменте DL-моделей **TabNet** и **TabNet + Heroes** демонстрируют сопоставимые результаты с ML-моделями, однако MLP-PyTorch достигает рекордного recall (0.771).
- DL модели могут быть особенно полезны, когда приоритетом является максимизация recall, но для сбалансированной работы по всем метрикам традиционные ML-решения остаются более надёжными.
