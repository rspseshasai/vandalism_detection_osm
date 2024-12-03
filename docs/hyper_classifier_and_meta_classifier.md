In this context, **hyper classifier** and **meta classifier** refer to similar overarching concepts but are distinguished by their specific implementation and roles within the machine learning pipeline. Let’s break them down:

---

### **Hyper Classifier**
- **Definition:** A **hyper classifier** is a model that is trained using aggregated predictions (or outputs) from other models as features. Here, it is specifically designed to process contributions within a changeset and predict vandalism at the changeset level using statistical summaries (aggregates) derived from contribution-level predictions.
- **Key Characteristics in This Context:**
  - Aggregates **per-contribution predictions** into a fixed-size feature vector for each changeset.
  - Uses statistical aggregates (e.g., mean, median, min, max, proportion of predictions, etc.) as its input features.
  - Operates at the **changeset level**, where the prediction is based on the combined contribution-level data.
  - It is **not integrated** with other features from the changeset model directly.
- **Purpose:** To leverage the **predictions** from contribution-level models to refine and enhance the accuracy of predictions for changesets.
- **Scope of Features:** Uses only aggregated predictions from contributions as its input, ignoring changeset-specific features (like timestamps, editor used, etc.).
- **Current Status in Your Pipeline:** 
  - You are training the hyper classifier using features derived solely from contribution-level predictions.

---

### **Meta Classifier**
- **Definition:** A **meta classifier** is a broader term that refers to any classifier that takes the outputs (predictions or probabilities) from one or more base classifiers and combines them to produce a final decision. 
- **Key Characteristics in This Context:**
  - Combines the predictions (or probabilities) from **multiple models** (e.g., the hyper classifier and the original changeset classifier).
  - Can integrate **changeset features** along with aggregated predictions and contributions.
  - Often used in **stacked generalization** or **ensemble learning**, where it acts as a second-layer model.
- **Purpose:** To combine the **predictions from both models** (the hyper classifier and the original changeset model) for improved overall performance.
- **Scope of Features:** 
  - Can include:
    - Predictions from the changeset classifier (based on changeset-level features).
    - Predictions from the hyper classifier (based on aggregated contribution predictions).
    - Changeset-level features (e.g., timestamps, editor used, geographic region, etc.).
- **Potential Future Use in Your Pipeline:**
  - Once both the changeset model and hyper classifier are trained and evaluated, you could combine their outputs using a meta classifier to further refine the prediction accuracy.

---

### **Comparison in Your Pipeline**

| **Aspect**             | **Hyper Classifier**                         | **Meta Classifier**                         |
|-------------------------|----------------------------------------------|---------------------------------------------|
| **Input**              | Aggregated features from contribution predictions. | Predictions from changeset and hyper classifiers, possibly with additional features. |
| **Purpose**            | Predict vandalism using contribution-level predictions. | Combine predictions from multiple models for better overall performance. |
| **Features**           | Statistical aggregates (mean, std, etc.).    | Predictions, probabilities, and potentially raw features. |
| **Level of Operation** | Changeset level, based on contribution predictions. | Post-processing level, combining models. |
| **Integration**        | Operates independently of the original changeset classifier. | Integrates outputs of the changeset model and hyper classifier. |
| **Use Case**           | Used to enhance changeset-level predictions without changeset features. | Used to combine all available models and features for final predictions. |

---

### **Pipeline Integration**
- **Current State:**
  - You have a **changeset model** predicting vandalism based on changeset features.
  - You have a **hyper classifier** predicting vandalism based on contribution-level aggregated predictions.
- **Future State (with Meta Classifier):**
  - After both models are trained, you can introduce a **meta classifier** that combines:
    - Predictions or probabilities from the **changeset model**.
    - Predictions or probabilities from the **hyper classifier**.
    - Possibly other features (e.g., original changeset features).

---

### **Key Difference**
The **hyper classifier** is a specific implementation focused on aggregating contribution-level predictions for changesets, while the **meta classifier** is a more general concept that combines the outputs of multiple models to improve overall predictions.

In your context:
1. The **hyper classifier** improves predictions for changesets by summarizing contribution-level data.
2. A potential **meta classifier** would combine both the hyper classifier and the original changeset model to create a more comprehensive prediction system.