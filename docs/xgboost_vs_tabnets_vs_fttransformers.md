### Model Comparison for Your Problem

Given that your dataset has around 40k entries with 40 features, and considering the nature of your binary
classification task (vandalism detection in OSM using tabular data), here’s a detailed comparison between **XGBoost**, *
*TabNet**, and **FT-Transformer** based on recent research:

---

#### **1. XGBoost**

**Overview**: XGBoost is a gradient-boosted decision tree (GBDT) model that has become the gold standard for tabular
data. It focuses on speed and performance optimization and is well-suited for small to medium-sized datasets like yours.

**Suitability**:

- **Data Compatibility**: It handles a variety of data formats, including numerical and categorical features, making it
  highly adaptable for your diverse features like changeset, user, and edit metrics.
- **Training Time**: It’s computationally efficient, even on smaller datasets, and can easily run on your 40k records.
  XGBoost can be further accelerated by using GPU support if needed.
- **Inference Time**: XGBoost is fast at inference, which is an advantage for large-scale applications and real-time
  systems.
- **Complexity**: It has lower complexity compared to deep learning models like FT-Transformer. Typically, you’ll
  experience linear scalability, making it ideal for datasets of your size.

**Research**: According to recent studies, XGBoost consistently outperforms deep learning models on small to
medium-sized tabular datasets【130†source】【131†source】. For your classification problem, its ability to rank feature
importance also offers explainability, which could be critical for justifying decisions in detecting vandalism.

---

#### **2. TabNet**

**Overview**: TabNet is a deep learning model explicitly designed for tabular data, using sparse attention to select
features dynamically. It’s more interpretable than many deep models, which is one of its strong suits.

**Suitability**:

- **Data Compatibility**: Like XGBoost, it can handle mixed types of tabular data. Its attention-based feature selection
  could potentially help in picking out the most important features in your dataset.
- **Training Time**: TabNet tends to require longer training times compared to XGBoost. While it can handle datasets of
  your size, the performance gains are often not significant unless the dataset is much larger.
- **Inference Time**: Inference is relatively slower than XGBoost due to its complex architecture, but still feasible
  for datasets of your size.
- **Complexity**: TabNet has a higher computational cost and may not show substantial performance improvements for 40k
  rows compared to simpler models like XGBoost.

**Research**: While TabNet shows promise on large and complex datasets, for smaller tabular datasets, its performance is
often not superior to tree-based methods. Research shows it can be overkill for small to mid-sized tabular
problems【130†source】【131†source】.

---

#### **3. FT-Transformer**

**Overview**: FT-Transformer is a transformer model adapted for tabular data, utilizing feature tokenization followed by
a transformer encoder. It’s a more recent approach designed to bridge the gap between tree-based models and deep
learning.

**Suitability**:

- **Data Compatibility**: It can effectively model both numerical and categorical data. It performs well when there are
  complex interactions between features, such as in your dataset with multiple types of features.
- **Training Time**: FT-Transformer requires significant training time and computational resources, more than XGBoost
  but less than TabNet when hyperparameters are well-tuned.
- **Inference Time**: Inference time is higher than XGBoost but still manageable with small to medium datasets. However,
  for your dataset size, the overhead might not be justified.
- **Complexity**: FT-Transformer is more complex than tree-based models, especially in terms of memory and computational
  requirements, which may not lead to significant gains given the relatively small size of your dataset.

**Research**: Studies show that FT-Transformers outperform both tree-based models and other deep learning methods on
some tabular datasets. However, this advantage becomes more pronounced on larger datasets with complex
interactions【130†source】【131†source】. For smaller datasets, its performance gain over XGBoost may not justify the
additional complexity.

---

### **Conclusion: Best Model for Your Problem**

For your specific use case (binary classification of vandalism detection in OSM), **XGBoost** is likely the most
suitable choice based on the following reasons:

1. **Efficiency**: XGBoost offers faster training and inference times, especially for datasets with around 40k entries.
2. **Performance**: It has consistently shown strong performance in classification tasks involving tabular data.
3. **Explainability**: XGBoost’s feature importance rankings can provide interpretable insights, useful for
   understanding which features contribute most to vandalism detection.
4. **Low Complexity**: The model's complexity and resource requirements are well-suited for your dataset size, while
   deep learning models like TabNet and FT-Transformers may not yield significantly better results in this case.

If you're exploring cutting-edge methods, you could still experiment with **FT-Transformer**, especially if you scale
your dataset or need to capture complex feature interactions. However, XGBoost is likely to give you the best trade-off
between performance and practicality for this task.

