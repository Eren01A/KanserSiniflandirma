import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

# GUI ile dosya seçimi
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="CSV Dosyasını Seç", filetypes=[("CSV files", "*.csv")])

# Dosyayı yükleme ve boş sütunu kaldırma
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["Unnamed: 32", "id"], errors='ignore')
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data

data = load_and_preprocess_data(file_path)

# Verilerin analiz edilmesi
def analyze_data(data):
    print("Veri Setinin İlk 5 Satırı:\n", data.head())
    print("\nTeşhis Dağılımı:\n", data["diagnosis"].value_counts())

analyze_data(data)

# Özellik ve hedef ayrımı
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# PCA uygulama
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA ile Açıklanan Varyans Oranları: {pca.explained_variance_ratio_}")
    return X_pca

X_pca = apply_pca(X)

# PCA sonuçlarını görselleştirme
def plot_pca(X_pca, y):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label="Teşhis")
    plt.title("PCA Sonuçları")
    plt.xlabel("1. Bileşen")
    plt.ylabel("2. Bileşen")
    plt.show()

plot_pca(X_pca, y)

# K-Means Clustering
def apply_kmeans(X_pca, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='coolwarm', alpha=0.8)
    plt.colorbar(scatter, label="Küme")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='X', label='Merkezler')
    plt.title("K-Means Kümeleme Sonuçları")
    plt.xlabel("1. Bileşen")
    plt.ylabel("2. Bileşen")
    plt.legend()
    plt.show()
    return clusters

clusters = apply_kmeans(X_pca)

# Veriyi eğitim ve test olarak ayırma
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(X_pca, y)

# Random Forest Modeli oluşturma ve eğitme
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

rf_model = train_random_forest(X_train, y_train)

# KNN Modeli oluşturma ve eğitme
def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

knn_model = train_knn(X_train, y_train)

# Tahmin yapma ve değerlendirme
def predict_and_evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Model Doğruluğu:", accuracy)
    print(f"\n{model_name} Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
    return y_pred, y_pred_prob

rf_y_pred, rf_y_pred_prob = predict_and_evaluate(rf_model, X_test, y_test, "Random Forest")
knn_y_pred, knn_y_pred_prob = predict_and_evaluate(knn_model, X_test, y_test, "KNN")

# Karışıklık matrisi görselleştirilmesi
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["İyi Huylu (B)", "Kötü Huylu (M)"], yticklabels=["İyi Huylu (B)", "Kötü Huylu (M)"])
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title(f"{model_name} Karışıklık Matrisi")
    plt.show()

plot_confusion_matrix(y_test, rf_y_pred, "Random Forest")
plot_confusion_matrix(y_test, knn_y_pred, "KNN")

# ROC Eğrisi çizimi
def plot_roc_curve(y_test, y_pred_prob, model_name):
    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"{model_name} AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("Yanlış Pozitif Oranı")
        plt.ylabel("Doğru Pozitif Oranı")
        plt.title(f"{model_name} ROC Eğrisi")
        plt.legend(loc="lower right")
        plt.show()

plot_roc_curve(y_test, rf_y_pred_prob, "Random Forest")
plot_roc_curve(y_test, knn_y_pred_prob, "KNN")

# Sonuçların görselleştirilmesi
def plot_diagnosis_distribution(data):
    counts = data["diagnosis"].value_counts(normalize=True) * 100
    labels = ["İyi Huylu (B)", "Kötü Huylu (M)"]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title("Hücrelerin İyi Huylu ve Kötü Huylu Dağılımı")
    plt.ylabel("Yüzde")
    plt.show()

plot_diagnosis_distribution(data)

# Özelliklerin önemini görselleştirme
def plot_feature_importance(model, X):
    importance = model.feature_importances_
    features = [f"PCA Bileşen {i+1}" for i in range(X.shape[1])]
    sorted_idx = importance.argsort()
    plt.figure(figsize=(10, 8))
    plt.barh([features[i] for i in sorted_idx], importance[sorted_idx], color='blue')
    plt.title("Özellik Önem Düzeyleri")
    plt.xlabel("Önem Skoru")
    plt.show()

plot_feature_importance(rf_model, X_pca)

# Kullanıcı için hücrelerin yüzde hesaplaması
def calculate_percentages(data):
    counts = data["diagnosis"].value_counts()
    total = len(data)
    benign_percentage = (counts[0] / total) * 100
    malignant_percentage = (counts[1] / total) * 100
    print(f"İyi Huylu Hücreler: %{benign_percentage:.2f}")
    print(f"Kötü Huylu Hücreler: %{malignant_percentage:.2f}")

calculate_percentages(data)

# Sonuçları dosyaya kaydetme
def save_results_to_file(data, file_name="results.csv"):
    data.to_csv(file_name, index=False)
    print(f"Sonuçlar {file_name} dosyasına kaydedildi.")

save_results_to_file(data)
