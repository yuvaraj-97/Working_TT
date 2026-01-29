# =============================================================================
# SVM & TF-IDF Vectorization - Step-by-Step Demo
# =============================================================================
# This script walks through:
#   1. How TF-IDF vectorization works (with a common-word example)
#   2. How SVM classification works (with a visual 2D plot)
# =============================================================================

# %% [Step 0] - Install/Import Libraries
# Run this cell first to make sure everything is available

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Change to 'Agg' if no display available

print("All libraries imported successfully!")


# =====================================================================
# PART 1 — TF-IDF VECTORIZATION
# =====================================================================

# %% [Step 1] - Two sentences with a COMMON word
# Notice both sentences contain the word "VALVE"

sentence_1 = "VALVE EXPANSION THERMOSTATIC"
sentence_2 = "VALVE BALL BRASS"

documents = [sentence_1, sentence_2]

print("=== Our Two Sentences ===")
print(f"Sentence 1: {sentence_1}")
print(f"Sentence 2: {sentence_2}")
print(f"\nCommon word: 'VALVE' (appears in BOTH sentences)")


# %% [Step 2] - What is TF-IDF?
# TF  = Term Frequency      → How often a word appears in THIS document
# IDF = Inverse Doc Frequency → How RARE the word is across ALL documents
#
# Key insight:
#   - Common words (like "VALVE") get LOWER scores
#   - Unique words (like "THERMOSTATIC", "BRASS") get HIGHER scores

print("\n=== TF-IDF Formula ===")
print("TF(word)  = (# times word appears in document) / (total words in document)")
print("IDF(word) = log( (total documents + 1) / (documents containing word + 1) ) + 1")
print("TF-IDF    = TF × IDF  (then normalized to unit length)")


# %% [Step 3] - Fit the TF-IDF Vectorizer (same as your SVM.py pipeline)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Show the vocabulary (each word gets a column index)
print("\n=== Vocabulary (word → column index) ===")
vocab = vectorizer.vocabulary_
for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print(f"  '{word}' → column {idx}")


# %% [Step 4] - See the actual TF-IDF vectors

print("\n=== TF-IDF Vectors ===")
feature_names = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=["Sentence 1", "Sentence 2"]
)
print(df_tfidf.round(4).to_string())


# %% [Step 5] - KEY OBSERVATION: The common word "valve" has LOWER score

print("\n=== Key Observation ===")
valve_score_s1 = df_tfidf.loc["Sentence 1", "valve"]
valve_score_s2 = df_tfidf.loc["Sentence 2", "valve"]
unique_word_s1 = df_tfidf.loc["Sentence 1", "thermostatic"]
unique_word_s2 = df_tfidf.loc["Sentence 2", "brass"]

print(f"'valve' score in Sentence 1:        {valve_score_s1:.4f}")
print(f"'valve' score in Sentence 2:        {valve_score_s2:.4f}")
print(f"'thermostatic' score in Sentence 1: {unique_word_s1:.4f}  ← HIGHER (unique word)")
print(f"'brass' score in Sentence 2:        {unique_word_s2:.4f}  ← HIGHER (unique word)")
print()
print("→ 'valve' appears in BOTH documents, so TF-IDF gives it a LOWER weight.")
print("→ 'thermostatic' and 'brass' are unique to their documents, so they get HIGHER weight.")
print("→ This is how the model learns to distinguish items that share common words!")


# %% [Step 6] - What happens with MORE documents?
# Adding more sentences to show how IDF changes

print("\n\n=== Adding More Documents ===")
documents_expanded = [
    "VALVE EXPANSION THERMOSTATIC",
    "VALVE BALL BRASS",
    "VALVE SOLENOID REFRIGERANT",
    "MOTOR BLOWER DIRECT DRIVE",
    "FILTER DRIER LIQUID LINE",
]

vectorizer2 = TfidfVectorizer()
tfidf_matrix2 = vectorizer2.fit_transform(documents_expanded)

feature_names2 = vectorizer2.get_feature_names_out()
df_tfidf2 = pd.DataFrame(
    tfidf_matrix2.toarray(),
    columns=feature_names2,
    index=[f"Doc {i+1}" for i in range(len(documents_expanded))]
)

print("Documents:")
for i, doc in enumerate(documents_expanded):
    print(f"  Doc {i+1}: {doc}")

print(f"\n=== TF-IDF Scores (showing 'valve' column) ===")
if 'valve' in df_tfidf2.columns:
    print(df_tfidf2[['valve']].round(4).to_string())
    print(f"\n→ 'valve' appears in 3 out of 5 documents now.")
    print(f"→ Its score dropped further because it's even MORE common across documents!")

print(f"\n=== Full TF-IDF Matrix ===")
print(df_tfidf2.round(4).to_string())


# =====================================================================
# PART 2 — HOW SVM CLASSIFICATION WORKS (Visual Example)
# =====================================================================

# %% [Step 7] - Create a simple 2D dataset to VISUALIZE SVM

print("\n\n" + "=" * 60)
print("PART 2: HOW SVM WORKS — Visual Example")
print("=" * 60)

# Imagine we have parts described by 2 features:
#   X-axis = TF-IDF score for "thermostatic" type words
#   Y-axis = TF-IDF score for "electrical" type words
#
# Category A = "Controls\\Valves"        (high thermostatic, low electrical)
# Category B = "Electrical Parts"        (low thermostatic, high electrical)

np.random.seed(42)

# Category A: Valves (cluster in bottom-right area)
X_valves = np.random.randn(20, 2) + [2, -1]
# Category B: Electrical (cluster in top-left area)
X_electrical = np.random.randn(20, 2) + [-1, 2]

X_train = np.vstack([X_valves, X_electrical])
y_train = np.array([0]*20 + [1]*20)  # 0 = Valves, 1 = Electrical

print("Training data created:")
print(f"  Category 0 (Valves):     {sum(y_train==0)} samples")
print(f"  Category 1 (Electrical): {sum(y_train==1)} samples")


# %% [Step 8] - Train the SVM (same kernel='linear' as your SVM.py)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

print("\nSVM Model trained with linear kernel (same as your pipeline)")
print(f"Support vectors found: {len(svm_model.support_vectors_)}")


# %% [Step 9] - Visualize the SVM decision boundary

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a mesh grid to plot the decision boundary
x_min, x_max = X_train[:, 0].min() - 1.5, X_train[:, 0].max() + 1.5
y_min, y_max = X_train[:, 1].min() - 1.5, X_train[:, 1].max() + 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                      np.linspace(y_min, y_max, 300))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
ax.contourf(xx, yy, Z, levels=[-10, 0, 10], alpha=0.15,
            colors=['#FF9999', '#9999FF'])
ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'],
           colors=['red', 'black', 'blue'], linewidths=[1, 2, 1])

# Plot data points
ax.scatter(X_valves[:, 0], X_valves[:, 1],
           c='red', marker='o', s=100, edgecolors='black',
           label='Valves (Category 0)', zorder=5)
ax.scatter(X_electrical[:, 0], X_electrical[:, 1],
           c='blue', marker='s', s=100, edgecolors='black',
           label='Electrical Parts (Category 1)', zorder=5)

# Highlight support vectors
sv = svm_model.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1],
           s=250, facecolors='none', edgecolors='green', linewidths=2,
           label=f'Support Vectors ({len(sv)})', zorder=6)

# Add a new test point
test_point = np.array([[0.5, 0.5]])
pred = svm_model.predict(test_point)
prob = svm_model.predict_proba(test_point)
pred_label = "Valves" if pred[0] == 0 else "Electrical"
ax.scatter(test_point[0, 0], test_point[0, 1],
           c='gold', marker='*', s=400, edgecolors='black', linewidths=1.5,
           label=f'New Item → Predicted: {pred_label} ({prob.max()*100:.1f}%)',
           zorder=7)

ax.set_xlabel('Feature 1 (e.g., "thermostatic" TF-IDF score)', fontsize=12)
ax.set_ylabel('Feature 2 (e.g., "electrical" TF-IDF score)', fontsize=12)
ax.set_title('SVM Classification — How It Separates Categories', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Decision Boundary\n(solid black line)',
            xy=(0.3, 0.8), fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
ax.annotate('Margin\n(dashed lines)',
            xy=(-0.8, -0.5), fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('SVM_Demo_Plot.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'SVM_Demo_Plot.png'")
plt.show()


# %% [Step 10] - Predict a new item & show confidence

print("\n=== Predicting a New Item ===")
print(f"Test point coordinates: {test_point[0]}")
print(f"Predicted category:     {pred_label}")
print(f"Confidence scores:      Valves={prob[0][0]*100:.1f}%, Electrical={prob[0][1]*100:.1f}%")

print("\n" + "=" * 60)
print("SUMMARY — How Your SVM Pipeline Works")
print("=" * 60)
print("""
1. PREPROCESSING
   → Extract First_Word from Item_Description
   → Group items by Partial_Taxonomy_Node

2. TF-IDF VECTORIZATION
   → Converts text descriptions into numerical vectors
   → Common words (like 'VALVE') get LOWER weight
   → Unique/descriptive words get HIGHER weight
   → This is what lets the model tell apart items with similar names

3. SVM TRAINING (per group)
   → Finds the best boundary (hyperplane) between categories
   → Support vectors = the critical data points near the boundary
   → Uses linear kernel for text classification

4. PREDICTION
   → New item description → TF-IDF vector → SVM predicts category
   → Confidence score tells how sure the model is
""")
