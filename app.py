import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- BAGIAN 1: KONFIGURASI & FUNGSI BANTUAN ---
# ==============================================================================

# Konfigurasi terpusat untuk nama file
# Kembali menggunakan dua file terpisah
FILE_DATA_UTAMA = "hasil_proses.csv"
FILE_PERBANDINGAN = "sentimen.csv"

# --- Fungsi-fungsi Pembantu (Helpers) ---

def plot_confusion_matrix(cm_data, labels, title):
    """Membuat dan mengembalikan figure plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    return fig

def plot_grouped_bar_chart(df, title, y_label, y_lim=(75, 85)):
    """Membuat dan mengembalikan figure grouped bar chart dari DataFrame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kernels = df['Kernel']
    metrics = df.columns[1:]
    x_pos = np.arange(len(kernels))
    width = 0.25
    colors = ['skyblue', 'orange', 'green']

    for i, metric in enumerate(metrics):
        offset = width * (i - 1)
        rects = ax.bar(x_pos + offset, df[metric], width, label=metric, color=colors[i])
        ax.bar_label(rects, padding=3, fmt='%.2f%%')

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kernels)
    ax.legend()
    ax.set_ylim(y_lim)
    ax.grid(axis='y', linestyle='--')
    return fig

def get_grid_search_text():
    """Mengembalikan teks hasil Grid Search."""
    return """
    Hasil Grid Search untuk Kernel Linear:
    Best Params: {'C': 1}
    Best Score: 0.788
    Accuracy: 81.38%
    Precision: 81.36%
    Recall: 81.38%

    Hasil Grid Search untuk Kernel RBF:
    Best Params: {'C': 10, 'gamma': 0.1}
    Best Score: 0.7896000000000001
    Accuracy: 82.12%
    Precision: 82.04%
    Recall: 82.12%

    Hasil Grid Search untuk Kernel Polynomial:
    Best Params: {'C': 1, 'degree': 1}
    Best Score: 0.788
    Accuracy: 81.38%
    Precision: 81.36%
    Recall: 81.38%
    """

def get_kfold_text():
    """Mengembalikan teks hasil K-Fold Cross Validation."""
    return """
Fold: 1, Akurasi: 83.80%, Presisi: 83.91%, Recall: 83.80%
Fold: 2, Akurasi: 82.68%, Presisi: 83.83%, Recall: 82.68%
Fold: 3, Akurasi: 79.89%, Presisi: 80.07%, Recall: 79.89%
Fold: 4, Akurasi: 81.01%, Presisi: 80.73%, Recall: 81.01%
Fold: 5, Akurasi: 74.30%, Presisi: 73.75%, Recall: 74.30%
Fold: 6, Akurasi: 81.56%, Presisi: 81.18%, Recall: 81.56%
Fold: 7, Akurasi: 87.15%, Presisi: 87.01%, Recall: 87.15%
Fold: 8, Akurasi: 79.78%, Presisi: 79.33%, Recall: 79.78%
Fold: 9, Akurasi: 78.09%, Presisi: 78.44%, Recall: 78.09%
Fold: 10, Akurasi: 77.53%, Presisi: 79.25%, Recall: 77.53%
=======================
Rata-rata Akurasi: 80.58%
Rata-rata Presisi: 80.75%
Rata-rata Recall: 80.58%
    
Fold dengan performa terbaik: 7
Akurasi terbaik: 87.15%
Presisi terbaik: 87.01%
Recall terbaik: 87.15%
    """

# ==============================================================================
# --- BAGIAN 2: TATA LETAK APLIKASI STREAMLIT ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Dashboard Hasil Analisis Sentimen")
st.title("Dashboard Visualisasi Hasil Analisis Sentimen SVM")

# --- Kontainer 1: Pemuatan Data Utama (DIPERBARUI) ---
with st.container(border=True):
    st.header("📁 Pemuatan Data Utama")
    st.write(f"Menampilkan cuplikan data utama (`{FILE_DATA_UTAMA}`) yang digunakan dalam analisis SVM.")
    try:
        # Membaca file hasil_proses.csv
        df_utama = pd.read_csv(FILE_DATA_UTAMA)
        st.dataframe(df_utama.head())
        st.info(f"Total data yang dianalisis: **{len(df_utama)} baris**.")
    except FileNotFoundError:
        st.error(f"File '{FILE_DATA_UTAMA}' tidak ditemukan. Silakan letakkan di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")


# --- Kontainer 2: Evaluasi Awal SVM ---
with st.container(border=True):
    st.header("📊 Tahap 1: Evaluasi Awal SVM")
    st.write("Hasil evaluasi awal sebelum penyetelan parameter.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi", "78.58%")
    col2.metric("Presisi", "81.38%")
    col3.metric("Recall", "78.58%")

    cm_initial_data = np.array([[83, 107], [8, 339]])
    st.pyplot(plot_confusion_matrix(cm_initial_data, ['negatif', 'positif'], "Confusion Matrix Tahap Awal"))

# --- Kontainer 3: Perbandingan Kernel ---
with st.container(border=True):
    st.header("⚙️ Tahap 2: Pencarian Hyperparameter & Perbandingan Kinerja Kernel")
    st.write("Parameter terbaik yang ditemukan untuk setiap kernel dan perbandingan kinerjanya.")
    
    st.write("#### Hasil Teks Grid Search:")
    st.code(get_grid_search_text())

    st.write("#### Grafik Perbandingan Kinerja Kernel")
    results_data = {
        'Kernel': ['Linear', 'RBF', 'Polynomial'],
        'Accuracy': [81.38, 82.12, 81.38],
        'Precision': [81.36, 82.04, 81.36],
        'Recall': [81.38, 82.12, 81.38]
    }
    results_df = pd.DataFrame(results_data)
    
    st.pyplot(plot_grouped_bar_chart(results_df, 'Perbandingan Kinerja Model SVM Berdasarkan Kernel', 'Performance (%)'))

# --- Kontainer 4: Hasil Akhir Terbaik ---
with st.container(border=True):
    st.header("🏆 Tahap 3: Visualisasi Hasil Terbaik (Kernel RBF, Rasio 90:10)")
    st.write("Hasil terbaik setelah diketahui kernel RBF adalah yang terbaik dan model dilatih kembali.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi", "84.35%")
    col2.metric("Presisi", "84.43%")
    col3.metric("Recall", "84.35%")

    cm_best_data = np.array([[40, 21], [7, 111]])
    st.pyplot(plot_confusion_matrix(cm_best_data, ['negatif', 'positif'], "Confusion Matrix Hasil Akhir Terbaik"))
    st.info("**Total Prediksi Benar:** 151 | **Total Keseluruhan Data:** 179")
    
# --- Kontainer 5: Distribusi Label (Lexicon vs. Prediksi) (DIPERBARUI) ---
with st.container(border=True):
    st.header("📊 Distribusi Label (Lexicon vs. Prediksi)")
    st.write(f"Membandingkan distribusi label dari metode leksikon dengan hasil prediksi model, berdasarkan file `{FILE_PERBANDINGAN}`.")
    try:
        # Memuat file 'sentimen.csv' secara spesifik di sini
        df_comparison = pd.read_csv(FILE_PERBANDINGAN, delimiter=';', header=1)
        
        st.write("Cuplikan Data untuk Grafik Distribusi:")
        st.dataframe(df_comparison.head())
        
        # Validasi kolom sebelum plotting
        required_cols = ['label_lexicon', 'label_prediksi']
        if not all(col in df_comparison.columns for col in required_cols):
            st.error(f"File '{FILE_PERBANDINGAN}' harus memiliki kolom: {', '.join(required_cols)}")
        else:
            st.write("Grafik Distribusi:")
            lexicon_counts = df_comparison['label_lexicon'].value_counts().sort_index()
            prediksi_counts = df_comparison['label_prediksi'].value_counts().sort_index()
            
            # Membuat plot
            fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
            x = np.arange(len(lexicon_counts.index))
            width = 0.35
            
            rects1 = ax_dist.bar(x - width/2, lexicon_counts.values, width, label='Lexicon', color='#1f77b4')
            rects2 = ax_dist.bar(x + width/2, prediksi_counts.values, width, label='Prediksi', color='#ff7f0e')
            
            ax_dist.set_ylabel('Jumlah')
            ax_dist.set_title('Distribusi Label')
            ax_dist.set_xticks(x, lexicon_counts.index)
            ax_dist.legend()
            
            for rect in rects1 + rects2:
                height = rect.get_height()
                ax_dist.annotate(f'{height}', xy=(rect.get_x() + rect.get_width()/2, height),
                                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            st.pyplot(fig_dist)

    except FileNotFoundError:
        st.error(f"File '{FILE_PERBANDINGAN}' tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat grafik distribusi: {e}")

# --- Kontainer 6: K-Fold Cross Validation ---
with st.container(border=True):
    st.header("🔁 Tahap 4: Hasil 10-Fold Cross Validation")
    st.write("Hasil pengujian konsistensi model terbaik menggunakan validasi silang 10-Fold.")
    st.code(get_kfold_text())