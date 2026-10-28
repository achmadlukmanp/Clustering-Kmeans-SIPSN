"""
Dashboard Pengelompokan Wilayah di Indonesia Berdasarkan Karakteristik
Pengelolaan Sampah Menggunakan K-Means.

Semua komputasi (baca file, agregasi multi-tahun, missing value,
standarisasi RobustScaler, K-Means, Silhouette Score, Davies-Bouldin Index)
dilakukan di backend Python (pandas + scikit-learn). Flask hanya mengatur
alur halaman & menyimpan state antar-langkah.
"""
import io
import re
import uuid

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                    session, flash)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
app.secret_key = "ganti-dengan-secret-key-anda-sendiri"


def get_metric_ks(n_samples):
    """Rentang k yang dievaluasi: 2 s.d. 6, tapi tidak boleh melebihi
    (n_samples - 1) karena Silhouette/DBI mensyaratkan jumlah label < jumlah sampel."""
    max_k = min(6, n_samples - 1)
    if max_k < 2:
        return []
    return list(range(2, max_k + 1))


# ------------------------------------------------------------------
# Server-side state store, per-session (demo: in-memory / single proses).
# Untuk produksi multi-worker, ganti dict ini dengan Redis / database.
# ------------------------------------------------------------------
STORE = {}


def get_state():
    sid = session.get("sid")
    if not sid or sid not in STORE:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        STORE[sid] = {}
    return STORE[sid]


def unlocked_map(state):
    return {
        "home": True,
        "upload": True,
        "processing": "working_df" in state,
        "modeling": "standardized" in state,
        "evaluation": "cluster_labels" in state,
        "interpretation": "cluster_labels" in state,
    }


CLUSTER_COLORS = ["#1F6F5C", "#D68C34", "#B65C38", "#4C5B67",
                  "#6B4E71", "#7C8B4A", "#2F6F8A", "#A13D63"]

# Parameter model FINAL yang benar-benar dipakai untuk k yang dipilih user
# (dipakai di modeling_run -> kartu skor di halaman Evaluasi -> tahap
# Interpretasi). Ini SUMBER KEBENARAN utama.
FINAL_N_INIT = 10
FINAL_RANDOM_STATE = 42

# Parameter model PEMBANDING untuk tabel evaluasi k=2..6 & grafik
# Elbow/Silhouette (compute_all_metrics). Disamakan PERSIS dengan notebook
# skripsi (KMeans(n_clusters=k, random_state=42, n_init=10) di loop
# K_range) supaya angka grafik & kartu skor model final selalu identik,
# bukan cuma "mirip" -- karena KMeans di sklearn deterministik untuk
# random_state & n_init yang sama pada data yang sama.
COMPARISON_N_INIT = 10
COMPARISON_RANDOM_STATE = 42


# ------------------------------------------------------------------
# Helper: baca file excel/csv -> DataFrame
# ------------------------------------------------------------------
def read_any(file_storage):
    filename = file_storage.filename
    ext = filename.rsplit(".", 1)[-1].lower()
    data = file_storage.read()
    buf = io.BytesIO(data)
    if ext == "csv":
        return pd.read_csv(buf)
    return pd.read_excel(buf, engine="openpyxl" if ext == "xlsx" else None)


def looks_numeric(series, sample=30):
    if pd.api.types.is_numeric_dtype(series):
        return True
    s = series.dropna().head(sample)
    if len(s) == 0:
        return False
    coerced = pd.to_numeric(s.astype(str).str.replace(".", "", regex=False)
                             .str.replace(",", ".", regex=False), errors="coerce")
    return coerced.notna().mean() > 0.7


def to_numeric_series(series):
    """Konversi kolom ke numerik, mendukung format Indonesia (1.234,5)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (series.astype(str)
               .str.strip()
               .str.replace(".", "", regex=False)
               .str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def auto_detect_geo_cols(headers):
    """Deteksi otomatis kolom Provinsi, Kabupaten/Kota, dan Tahun dari nama kolom.
    User tidak perlu memilih manual — cukup notifikasi bila terdeteksi multi-tahun."""
    prov_col = next((h for h in headers if re.search(r"provinsi", h, re.I)), None)
    kab_col = next((h for h in headers if re.search(r"kab|kota", h, re.I)), None)
    tahun_col = next((h for h in headers if re.search(r"tahun|year", h, re.I)), None)
    return prov_col, kab_col, tahun_col


def compute_outlier_bounds(df, columns):
    """Batas bawah/atas outlier per kolom memakai metode IQR (1.5 x IQR),
    persis seperti notebook skripsi: Q1 - 1.5*IQR s.d. Q3 + 1.5*IQR."""
    bounds = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[col] = (float(lower), float(upper))
    return bounds


def count_outliers(df, columns):
    """Jumlah baris di luar batas IQR, per kolom (fungsi jumlah_outlier di notebook)."""
    bounds = compute_outlier_bounds(df, columns)
    counts = {}
    for col in columns:
        lower, upper = bounds[col]
        counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def handle_outliers_iqr(df, columns):
    """Menangani outlier dengan CAPPING/WINSORIZING berbasis IQR: nilai yang
    lebih kecil dari lower_bound diganti lower_bound, nilai yang lebih besar
    dari upper_bound diganti upper_bound -- persis fungsi handle_outliers_iqr()
    pada notebook skripsi (bukan menghapus baris, hanya memangkas nilainya)."""
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df


def find_elbow_k(ks, sses):
    x1, y1, x2, y2 = ks[0], sses[0], ks[-1], sses[-1]
    best_k, best_d = ks[0], -1
    for k, sse in zip(ks, sses):
        num = abs((y2 - y1) * k - (x2 - x1) * sse + x2 * y1 - y2 * x1)
        den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        d = 0 if den == 0 else num / den
        if d > best_d:
            best_d, best_k = d, k
    return best_k


def silhouette_interp(score):
    if score >= 0.71:
        return "sangat baik", "struktur klaster sangat kuat"
    if score >= 0.51:
        return "baik", "struktur klaster cukup kuat"
    if score >= 0.26:
        return "cukup", "struktur klaster lemah namun masih dapat diterima"
    return "kurang baik", "tidak ditemukan struktur klaster yang jelas"


def dbi_interp(score):
    if score < 0.5:
        return "sangat baik", "klaster sangat rapat dan terpisah jelas satu sama lain"
    if score < 1.0:
        return "baik", "klaster cukup terpisah dengan baik"
    if score < 1.5:
        return "cukup", "pemisahan klaster cukup, ada sedikit tumpang tindih antar klaster"
    return "kurang baik", "klaster banyak tumpang tindih / kurang terpisah dengan baik"


def compute_all_metrics(X, ks):
    """Fit KMeans SEKALI per-k untuk keperluan PERBANDINGAN (tabel k=2..6,
    grafik Elbow & Silhouette). Bukan model final yang dipakai untuk hasil
    klaster akhir -- lihat modeling_run() untuk model final tersebut."""
    results = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=COMPARISON_N_INIT,
                    random_state=COMPARISON_RANDOM_STATE).fit(X)
        results[k] = {
            "sse": float(km.inertia_),
            "sil": float(silhouette_score(X, km.labels_)),
            "dbi": float(davies_bouldin_score(X, km.labels_)),
        }
    return results


def compute_k_optimal(metrics, ks):
    """K optimal berdasarkan KOMBINASI 2 metrik: Silhouette Score tertinggi
    DAN Davies-Bouldin Index terendah (rank gabungan kedua metrik)."""
    sil_rank = {k: r for r, k in enumerate(sorted(ks, key=lambda kk: -metrics[kk]["sil"]), start=1)}
    dbi_rank = {k: r for r, k in enumerate(sorted(ks, key=lambda kk: metrics[kk]["dbi"]), start=1)}
    combined = {k: sil_rank[k] + dbi_rank[k] for k in ks}
    return min(ks, key=lambda kk: (combined[kk], kk))


# ====================================================================
# 1. HOME
# ====================================================================
@app.route("/")
def home():
    state = get_state()
    return render_template("home.html", unlocked=unlocked_map(state))


# ====================================================================
# 2. UPLOAD DATA
# ====================================================================
@app.route("/upload", methods=["GET"])
def upload_page():
    state = get_state()
    ctx = {"unlocked": unlocked_map(state)}
    if "raw_df" not in state:
        return render_template("upload.html", **ctx)

    raw_df = state["raw_df"]
    ctx.update({
        "file_names": state.get("file_names", []),
        "headers": list(raw_df.columns),
        "preview_rows": raw_df.head(15).where(pd.notnull(raw_df.head(15)), None).to_dict("records"),
        "row_count": len(raw_df),
        "col_count": len(raw_df.columns),
        "multi_year": state.get("multi_year", False),
        "n_years": state.get("n_years"),
    })

    # kandidat variabel: hanya kolom numerik, di luar kolom Provinsi/Kab-Kota/Tahun.
    # Tersedia langsung setelah upload & pratinjau data (agregasi dipindah ke tahap Processing).
    exclude = {state.get("prov_col"), state.get("kab_col"), state.get("tahun_col")}
    var_headers = [c for c in raw_df.columns if c not in exclude and looks_numeric(raw_df[c])]
    ctx["var_headers"] = var_headers

    return render_template("upload.html", **ctx)


@app.route("/upload/merge", methods=["POST"])
def upload_merge():
    state = get_state()
    files = request.files.getlist("files")
    files = [f for f in files if f and f.filename]
    if not files:
        flash("Pilih minimal satu file terlebih dahulu.", "warn")
        return redirect(url_for("upload_page"))

    dfs, names = [], state.get("file_names", [])
    try:
        for f in files:
            dfs.append(read_any(f))
            names.append(f.filename)
    except Exception as e:
        flash(f"Gagal membaca file: {e}", "error")
        return redirect(url_for("upload_page"))

    prev = state.get("raw_df")
    all_dfs = ([prev] if prev is not None else []) + dfs
    merged = pd.concat(all_dfs, ignore_index=True, sort=False)

    state["raw_df"] = merged
    state["file_names"] = names

    # deteksi otomatis kolom Provinsi / Kabupaten-Kota / Tahun (tidak perlu input user)
    prov_col, kab_col, tahun_col = auto_detect_geo_cols(list(merged.columns))
    state["prov_col"] = prov_col
    state["kab_col"] = kab_col
    state["tahun_col"] = tahun_col
    if tahun_col:
        n_years = int(merged[tahun_col].nunique(dropna=True))
        state["multi_year"] = n_years > 1
        state["n_years"] = n_years
    else:
        state["multi_year"] = False
        state["n_years"] = None

    # reset tahap berikutnya karena data mentah berubah
    for k in ("selected_vars", "working_df", "imputed_df", "aggregated", "agg_row_count",
              "outlier_counts", "outlier_percents", "outlier_handled",
              "outlier_handled_counts", "outlier_handled_percents",
              "standardized", "cluster_labels", "final_silhouette", "final_dbi",
              "elbow_results", "silhouette_results", "metrics_cache", "chosen_k",
              "timbulan_col", "pengurangan_col", "penanganan_col"):
        state.pop(k, None)
    flash(f"{len(files)} file berhasil digabungkan. Total {len(merged)} baris.", "ok")
    return redirect(url_for("upload_page") + "#card-ringkasan")


@app.route("/upload/reset_files", methods=["POST"])
def upload_reset_files():
    state = get_state()
    state.clear()
    return redirect(url_for("upload_page"))


@app.route("/upload/proceed", methods=["POST"])
def upload_proceed():
    state = get_state()
    if "raw_df" not in state:
        flash("Unggah data terlebih dahulu.", "warn")
        return redirect(url_for("upload_page"))

    selected = request.form.getlist("selected_vars")
    if len(selected) < 2:
        flash("Pilih minimal 2 variabel numerik untuk diproses.", "warn")
        return redirect(url_for("upload_page"))

    raw_df = state["raw_df"]
    prov_col, kab_col = state.get("prov_col"), state.get("kab_col")

    working_df = raw_df[selected].copy()

    state["selected_vars"] = selected
    state["prov_values"] = raw_df[prov_col].tolist() if prov_col else None
    state["kab_values"] = raw_df[kab_col].tolist() if kab_col else None
    state["working_df"] = working_df
    state["rename_map"] = {c: c for c in selected}

    # Lacak identitas kolom Timbulan/Pengurangan/Penanganan dari nama ASLI (sebelum
    # rename), agar tetap bisa dipakai untuk hitung persentase meskipun user
    # nanti mengganti nama variabelnya sebebas apapun.
    state["timbulan_col"] = next((h for h in selected if re.search(r"timbul", h, re.I)), None)
    state["pengurangan_col"] = next((h for h in selected if re.search(r"kurang|gurang", h, re.I)), None)
    state["penanganan_col"] = next((h for h in selected if re.search(r"tangan|nangan", h, re.I)), None)

    for k in ("imputed_df", "aggregated", "agg_row_count", "standardized",
              "cluster_labels", "final_silhouette", "final_dbi", "elbow_results",
              "silhouette_results", "metrics_cache", "chosen_k", "renamed_done",
              "missing_counts", "handled_counts", "outlier_counts", "outlier_percents",
              "outlier_handled", "outlier_handled_counts", "outlier_handled_percents"):
        state.pop(k, None)

    return redirect(url_for("processing_page"))


# ====================================================================
# 3. PROCESSING DATA (hanya variabel numerik terpilih)
#    Rename -> Cek Missing Value -> Tangani Missing Value ->
#    Agregasi Data Multi-Tahun -> Standarisasi (RobustScaler)
# ====================================================================
@app.route("/processing", methods=["GET"])
def processing_page():
    state = get_state()
    if "working_df" not in state:
        flash("Selesaikan tahap Upload Data terlebih dahulu.", "warn")
        return redirect(url_for("upload_page"))

    df = state["working_df"]
    ctx = {
        "unlocked": unlocked_map(state),
        "headers": list(df.columns),
        "rename_map": state.get("rename_map", {c: c for c in df.columns}),
        "multi_year": state.get("multi_year", False),
        "n_years": state.get("n_years"),
        "aggregated": state.get("aggregated", False),
        "agg_row_count": state.get("agg_row_count"),
    }
    if state.get("renamed_done"):
        ctx["renamed_preview"] = df.head(10).where(pd.notnull(df.head(10)), None).to_dict("records")

    if "missing_counts" in state:
        ctx["missing_counts"] = state["missing_counts"]
        ctx["missing_total"] = sum(state["missing_counts"].values())

    if "handled_counts" in state:
        ctx["handled_counts"] = state["handled_counts"]
        ctx["handled_total"] = sum(state["handled_counts"].values())

    if state.get("aggregated") and "imputed_df" in state:
        agg_df = state["imputed_df"].round(2).copy()
        prov_col, kab_col = state.get("prov_col"), state.get("kab_col")
        # tampilkan kolom identitas wilayah (Provinsi/Kabupaten-Kota) di depan, supaya
        # tiap baris hasil agregasi bisa dicocokkan langsung dengan notebook/Colab.
        if kab_col and state.get("kab_values") and len(state["kab_values"]) == len(agg_df):
            agg_df.insert(0, kab_col, state["kab_values"])
        if prov_col and state.get("prov_values") and len(state["prov_values"]) == len(agg_df):
            agg_df.insert(0, prov_col, state["prov_values"])
        ctx["agg_preview"] = agg_df.head(10).where(pd.notnull(agg_df.head(10)), None).to_dict("records")
        ctx["agg_cols"] = list(agg_df.columns)
        ctx["agg_message"] = state.get("agg_message")

    ctx["outlier_handled"] = state.get("outlier_handled", False)
    if "outlier_counts" in state:
        ctx["outlier_counts"] = state["outlier_counts"]
        ctx["outlier_percents"] = state["outlier_percents"]
        ctx["outlier_total"] = sum(state["outlier_counts"].values())

    if state.get("outlier_handled"):
        ctx["outlier_handled_counts"] = state["outlier_handled_counts"]
        ctx["outlier_handled_percents"] = state["outlier_handled_percents"]
        ctx["outlier_handled_total"] = sum(state["outlier_handled_counts"].values())

    if "standardized" in state:
        std_arr = state["standardized"]
        cols = list(state["imputed_df"].columns)
        rows = []
        for i in range(len(std_arr)):
            row = {c: round(float(std_arr[i][j]), 3) for j, c in enumerate(cols)}
            rows.append(row)
        ctx["standardize_rows"] = rows
        ctx["standardize_cols"] = cols

    return render_template("processing.html", **ctx)


@app.route("/processing/rename", methods=["POST"])
def processing_rename():
    state = get_state()
    df = state["working_df"]
    new_map = {}
    for col in df.columns:
        new_name = request.form.get(f"rename__{col}", col).strip() or col
        new_map[col] = new_name
    df = df.rename(columns=new_map)
    state["working_df"] = df
    state["rename_map"] = {c: c for c in df.columns}
    state["renamed_done"] = True

    # ikuti perubahan nama pada kolom Timbulan/Pengurangan/Penanganan yang dilacak,
    # supaya identitasnya tidak hilang meskipun user mengganti namanya
    for key in ("timbulan_col", "pengurangan_col", "penanganan_col"):
        old_name = state.get(key)
        if old_name and old_name in new_map:
            state[key] = new_map[old_name]

    # reset tahap-tahap setelah rename
    for k in ("missing_counts", "handled_counts", "imputed_df", "aggregated",
              "agg_row_count", "agg_message", "outlier_counts", "outlier_percents",
              "outlier_handled", "outlier_handled_counts", "outlier_handled_percents",
              "standardized", "cluster_labels", "final_silhouette",
              "final_dbi", "elbow_results", "silhouette_results", "metrics_cache", "chosen_k"):
        state.pop(k, None)
    return redirect(url_for("processing_page") + "#card-rename")


@app.route("/processing/check_missing", methods=["POST"])
def processing_check_missing():
    """Menghitung jumlah nilai kosong/NaN pada variabel yang dipilih,
    berdasarkan data mentah (per baris asli hasil upload, sebelum agregasi)."""
    state = get_state()
    df = state["working_df"]
    counts = {}
    for col in df.columns:
        coerced = to_numeric_series(df[col])
        counts[col] = int(coerced.isna().sum())
    state["missing_counts"] = counts
    for k in ("handled_counts", "imputed_df", "aggregated", "agg_row_count", "agg_message",
              "outlier_counts", "outlier_percents", "outlier_handled",
              "outlier_handled_counts", "outlier_handled_percents",
              "standardized", "cluster_labels", "final_silhouette", "final_dbi",
              "elbow_results", "silhouette_results", "metrics_cache", "chosen_k"):
        state.pop(k, None)
    return redirect(url_for("processing_page") + "#card-missing")


@app.route("/processing/handle_missing", methods=["POST"])
def processing_handle_missing():
    """Menangani missing value dengan mengisi 0 pada data mentah/per-baris asli,
    persis seperti notebook: df_capaian = df_capaian.fillna(0)."""
    state = get_state()
    df = state["working_df"].copy()
    for col in df.columns:
        coerced = to_numeric_series(df[col])
        df[col] = coerced.fillna(0)
    state["imputed_df"] = df
    state["handled_counts"] = {col: int(df[col].isna().sum()) for col in df.columns}

    # jika data hanya 1 tahun, agregasi tidak diperlukan -> langsung tandai selesai
    state["aggregated"] = not state.get("multi_year", False)
    state["agg_row_count"] = len(df) if state["aggregated"] else None

    for k in ("outlier_counts", "outlier_percents", "outlier_handled",
              "outlier_handled_counts", "outlier_handled_percents",
              "standardized", "cluster_labels", "final_silhouette", "final_dbi",
              "elbow_results", "silhouette_results", "metrics_cache", "chosen_k"):
        state.pop(k, None)
    return redirect(url_for("processing_page") + "#card-handle")


@app.route("/processing/aggregate", methods=["POST"])
def processing_aggregate():
    """groupby(['Provinsi','Kabupaten/Kota']) lalu MEAN seluruh variabel numerik
    terpilih, sehingga tiap baris hasil mewakili rata-rata data satu Kabupaten/Kota
    dari seluruh tahun. PENTING: dihitung dari data yang missing value-nya SUDAH
    ditangani (diisi 0 pada tahap sebelumnya), bukan dari data mentah — persis
    seperti notebook:
        df_capaian = df_capaian.fillna(0)
        df_agg = df_capaian.groupby(["Provinsi","Kabupaten/Kota"])[[...]].mean()
    Jika di sini kita mengambil ulang data mentah lalu mengonversinya ke numerik
    lagi, baris yang tadinya kosong akan kembali menjadi NaN, dan groupby().mean()
    pandas akan MENGABAIKAN (skip) NaN tersebut alih-alih menghitungnya sebagai 0
    -- menghasilkan rata-rata, standardisasi, dan akhirnya Silhouette Score/DBI
    yang berbeda dari notebook."""
    state = get_state()
    if "imputed_df" not in state or "handled_counts" not in state:
        flash("Tangani missing value terlebih dahulu.", "warn")
        return redirect(url_for("processing_page"))

    prov_col, kab_col = state.get("prov_col"), state.get("kab_col")
    if not (prov_col and kab_col):
        flash("Kolom Provinsi/Kabupaten-Kota tidak dapat dideteksi otomatis dari data ini, "
              "sehingga agregasi tidak dapat dilakukan.", "warn")
        return redirect(url_for("processing_page"))

    numeric_cols = list(state["imputed_df"].columns)
    temp = state["imputed_df"].copy()
    temp[prov_col] = state.get("prov_values")
    temp[kab_col] = state.get("kab_values")

    agg = temp.groupby([prov_col, kab_col], as_index=False)[numeric_cols].mean()

    state["imputed_df"] = agg[numeric_cols]
    state["prov_values"] = agg[prov_col].tolist()
    state["kab_values"] = agg[kab_col].tolist()
    state["aggregated"] = True
    state["agg_row_count"] = len(agg)

    for k in ("outlier_counts", "outlier_percents", "outlier_handled",
              "outlier_handled_counts", "outlier_handled_percents",
              "standardized", "cluster_labels", "final_silhouette", "final_dbi",
              "elbow_results", "silhouette_results", "metrics_cache", "chosen_k"):
        state.pop(k, None)

    state["agg_message"] = f"Data berhasil diagregasi (Rata-rata/MEAN) menjadi {len(agg)} baris unik Kabupaten/Kota."
    return redirect(url_for("processing_page") + "#card-aggregate")


@app.route("/processing/check_outlier", methods=["POST"])
def processing_check_outlier():
    """Deteksi outlier tiap variabel numerik menggunakan metode IQR (1.5 x IQR),
    dihitung dari data yang SUDAH melalui tahap agregasi & penanganan missing
    value -- persis seperti notebook (dihitung dari df_agg)."""
    state = get_state()
    if "imputed_df" not in state or not state.get("aggregated"):
        flash("Agregasikan data terlebih dahulu.", "warn")
        return redirect(url_for("processing_page"))

    df = state["imputed_df"]
    cols = list(df.columns)
    counts = count_outliers(df, cols)
    total = len(df)
    percents = {c: round(counts[c] / total * 100, 2) if total else 0.0 for c in cols}

    state["outlier_counts"] = counts
    state["outlier_percents"] = percents
    for k in ("outlier_handled", "outlier_handled_counts", "outlier_handled_percents",
              "standardized", "cluster_labels", "final_silhouette", "final_dbi",
              "elbow_results", "silhouette_results", "metrics_cache", "chosen_k"):
        state.pop(k, None)
    return redirect(url_for("processing_page") + "#card-outlier")


@app.route("/processing/handle_outlier", methods=["POST"])
def processing_handle_outlier():
    """Menangani outlier dengan CAPPING berbasis IQR: nilai di luar
    [Q1-1.5*IQR, Q3+1.5*IQR] diganti dengan batas terdekat -- persis fungsi
    handle_outliers_iqr() pada notebook skripsi."""
    state = get_state()
    if "outlier_counts" not in state:
        flash("Cek outlier terlebih dahulu.", "warn")
        return redirect(url_for("processing_page"))

    df = state["imputed_df"]
    cols = list(df.columns)
    handled_df = handle_outliers_iqr(df, cols)
    state["imputed_df"] = handled_df

    counts_after = count_outliers(handled_df, cols)
    total = len(handled_df)
    percents_after = {c: round(counts_after[c] / total * 100, 2) if total else 0.0 for c in cols}

    state["outlier_handled"] = True
    state["outlier_handled_counts"] = counts_after
    state["outlier_handled_percents"] = percents_after

    for k in ("standardized", "cluster_labels", "final_silhouette", "final_dbi",
              "elbow_results", "silhouette_results", "metrics_cache", "chosen_k"):
        state.pop(k, None)
    return redirect(url_for("processing_page") + "#card-outlier")


@app.route("/processing/standardize", methods=["POST"])
def processing_standardize():
    state = get_state()
    if "imputed_df" not in state:
        flash("Tangani missing value terlebih dahulu.", "warn")
        return redirect(url_for("processing_page"))
    if not state.get("aggregated"):
        flash("Agregasikan data terlebih dahulu (data mencakup lebih dari 1 tahun).", "warn")
        return redirect(url_for("processing_page"))
    if not state.get("outlier_handled"):
        flash("Tangani outlier terlebih dahulu.", "warn")
        return redirect(url_for("processing_page"))
    df = state["imputed_df"]
    scaler = RobustScaler()  # (x - median) / IQR, tahan terhadap outlier
    std_arr = scaler.fit_transform(df.values)
    state["standardized"] = std_arr
    state["scaler_center"] = scaler.center_.tolist()
    state["scaler_scale"] = scaler.scale_.tolist()

    # PENTING: setiap kali standarisasi dijalankan ULANG, data numerik yang
    # masuk ke KMeans bisa berubah (mis. user mengulang tahap sebelumnya lalu
    # klik "Standarisasi Data" lagi). Semua cache/hasil pemodelan yang dihitung
    # dari array standardized LAMA wajib dibuang di sini -- kalau tidak,
    # grafik Elbow/Silhouette di /modeling akan menampilkan angka basi dari
    # metrics_cache lama, sementara /evaluation (yang selalu fit ulang model
    # final di modeling_run) menampilkan angka yang benar/fresh -- inilah
    # penyebab dua halaman terlihat "tidak sinkron".
    for k in ("metrics_cache", "elbow_results", "silhouette_results",
              "cluster_labels", "final_silhouette", "final_dbi", "chosen_k"):
        state.pop(k, None)

    return redirect(url_for("processing_page") + "#card-standardize")


# ====================================================================
# 4. PEMODELAN K-MEANS (k = 2 s.d. 6)
# ====================================================================
@app.route("/modeling", methods=["GET"])
def modeling_page():
    state = get_state()
    if "standardized" not in state:
        flash("Selesaikan tahap Processing Data (standarisasi) terlebih dahulu.", "warn")
        return redirect(url_for("processing_page"))
    X = state["standardized"]
    max_k = min(6, len(X) - 1)
    ks = get_metric_ks(len(X))

    # Hitung (atau pakai cache yang sudah dihitung sebelumnya untuk X yang SAMA
    # -- cache ini otomatis dibuang di processing_standardize() setiap kali
    # standarisasi dijalankan ulang, jadi selalu sinkron dengan data terbaru)
    # supaya grafik Elbow & Silhouette menampilkan angka ASLI, bukan dummy.
    if "metrics_cache" not in state:
        state["metrics_cache"] = compute_all_metrics(X, ks)
    metrics = state["metrics_cache"]

    sse_values = [round(metrics[k]["sse"], 2) for k in ks]
    sil_values = [round(metrics[k]["sil"], 4) for k in ks]
    chart_labels = [f"k={k}" for k in ks]

    ctx = {
        "unlocked": unlocked_map(state),
        "chosen_k": min(state.get("chosen_k", 3), max_k),
        "max_k": max_k,
        "k_marks": ks,
        "sse_values": sse_values,
        "sil_values": sil_values,
        "chart_labels": chart_labels,
    }
    return render_template("modeling.html", **ctx)


@app.route("/modeling/elbow", methods=["POST"])
def modeling_elbow():
    state = get_state()
    X = state["standardized"]
    ks = get_metric_ks(len(X))
    if "metrics_cache" not in state:
        state["metrics_cache"] = compute_all_metrics(X, ks)
    metrics = state["metrics_cache"]
    sses = [metrics[k]["sse"] for k in ks]
    elbow_k = find_elbow_k(ks, sses)
    state["elbow_results"] = {"ks": ks, "sses": sses, "elbow_k": elbow_k}
    return redirect(url_for("modeling_page") + "#card-elbow")


@app.route("/modeling/silhouette", methods=["POST"])
def modeling_silhouette():
    state = get_state()
    X = state["standardized"]
    ks = get_metric_ks(len(X))
    if "metrics_cache" not in state:
        state["metrics_cache"] = compute_all_metrics(X, ks)
    metrics = state["metrics_cache"]
    sils = [metrics[k]["sil"] for k in ks]
    best_k = ks[int(np.argmax(sils))]
    state["silhouette_results"] = {"ks": ks, "sils": sils, "best_k": best_k}
    return redirect(url_for("modeling_page") + "#card-silhouette")


@app.route("/modeling/run", methods=["POST"])
def modeling_run():
    """Fit model FINAL untuk k yang dipilih user. Inilah SATU-SATUNYA model
    yang dipakai untuk: kartu skor di halaman Evaluasi, baris tabel k terpilih,
    dan seluruh tahap Interpretasi Klaster berikutnya. Parameter (FINAL_N_INIT,
    FINAL_RANDOM_STATE) sengaja dipisah dari compute_all_metrics() yang hanya
    dipakai untuk tabel perbandingan k lain / grafik Elbow-Silhouette."""
    state = get_state()
    X = state["standardized"]
    max_k = min(6, len(X) - 1)
    k = int(request.form.get("k", 3))
    k = max(2, min(max_k, k))

    km = KMeans(n_clusters=k, n_init=FINAL_N_INIT,
                random_state=FINAL_RANDOM_STATE).fit(X)
    labels = km.labels_.tolist()
    sil = float(silhouette_score(X, km.labels_))
    dbi = float(davies_bouldin_score(X, km.labels_))

    state["chosen_k"] = k
    state["cluster_labels"] = labels
    state["final_silhouette"] = sil
    state["final_dbi"] = dbi
    return redirect(url_for("evaluation_page"))


# ====================================================================
# 5. EVALUASI MODEL (Silhouette Score + Davies-Bouldin Index, k = 2 s.d. 6)
# ====================================================================
@app.route("/evaluation", methods=["GET"])
def evaluation_page():
    state = get_state()
    if "cluster_labels" not in state:
        flash("Jalankan pemodelan K-Means terlebih dahulu.", "warn")
        return redirect(url_for("modeling_page"))

    chosen_k = state["chosen_k"]
    score = state["final_silhouette"]
    dbi = state["final_dbi"]
    label, detail = silhouette_interp(score)
    dbi_label, dbi_detail = dbi_interp(dbi)
    pct = max(0, min(100, ((score + 1) / 2) * 100))
    dbi_pct = max(0, min(100, 100 - dbi * 40))

    X = state["standardized"]
    ks = get_metric_ks(len(X))
    if "metrics_cache" not in state:
        state["metrics_cache"] = compute_all_metrics(X, ks)
    metrics = state["metrics_cache"]

    # PENTING: baris tabel untuk k yang SEDANG DIPILIH user (chosen_k) harus
    # menampilkan angka yang SAMA PERSIS dengan kartu skor di atas, karena
    # kartu berasal dari model final yang benar-benar dipakai (lihat
    # modeling_run). Baris k lain tetap memakai metrics_cache sebagai
    # perbandingan/referensi tren antar-k saja.
    metrics_for_table = {kk: dict(v) for kk, v in metrics.items()}
    if chosen_k in metrics_for_table:
        metrics_for_table[chosen_k]["sil"] = score
        metrics_for_table[chosen_k]["dbi"] = dbi

    table_rows = [{"k": kk, "sil": round(metrics_for_table[kk]["sil"], 3),
                   "dbi": round(metrics_for_table[kk]["dbi"], 3)} for kk in ks]

    # k optimal & angka pada kalimat "Kesimpulan" dihitung dari
    # metrics_for_table yang sudah disinkronkan dengan model final di atas,
    # sehingga jika chosen_k kebetulan juga k_optimal, angkanya tetap
    # konsisten dengan kartu.
    k_optimal = compute_k_optimal(metrics_for_table, ks)
    k_optimal_sil = round(metrics_for_table[k_optimal]["sil"], 3)
    k_optimal_dbi = round(metrics_for_table[k_optimal]["dbi"], 3)

    return render_template(
        "evaluation.html",
        unlocked=unlocked_map(state),
        score=score, pct=pct, k=chosen_k,
        label=label, detail=detail,
        dbi=dbi, dbi_label=dbi_label, dbi_detail=dbi_detail, dbi_pct=dbi_pct,
        table_rows=table_rows, k_optimal=k_optimal,
        k_optimal_sil=k_optimal_sil, k_optimal_dbi=k_optimal_dbi,
    )


# ====================================================================
# 6. INTERPRETASI KLASTER
# ====================================================================
@app.route("/interpretation", methods=["GET"])
def interpretation_page():
    state = get_state()
    if "cluster_labels" not in state:
        flash("Selesaikan tahap Evaluasi Model terlebih dahulu.", "warn")
        return redirect(url_for("evaluation_page"))

    k = state["chosen_k"]
    labels = np.array(state["cluster_labels"])
    imputed = state["imputed_df"].reset_index(drop=True)
    headers = list(imputed.columns)
    n = len(labels)

    counts = [int((labels == c).sum()) for c in range(k)]
    pie_labels = [f"Klaster {c+1} ({counts[c]/n*100:.1f}%)" for c in range(k)]

    prov_table = None
    if state.get("prov_col"):
        prov_series = pd.Series(state["prov_values"]).fillna("(tidak diketahui)")
        crosstab = pd.crosstab(prov_series, labels)
        crosstab = crosstab.reindex(columns=range(k), fill_value=0)
        crosstab["Total"] = crosstab.sum(axis=1)
        crosstab = crosstab.sort_values("Total", ascending=False)  # tertinggi -> terendah

        # detail wilayah (kab/kota) per provinsi per klaster, untuk tombol "Detail Wilayah"
        kab_values_for_detail = state.get("kab_values")
        prov_list = prov_series.tolist()
        detail = {}
        for i in range(n):
            p = prov_list[i]
            c = int(labels[i])
            name = str(kab_values_for_detail[i]) if kab_values_for_detail and kab_values_for_detail[i] is not None else f"Baris {i+1}"
            detail.setdefault(p, {}).setdefault(c, []).append(name)

        prov_table = {"provinces": crosstab.index.tolist(), "rows": crosstab.values.tolist(), "detail": detail}

    kab_lists = []
    kab_values = state.get("kab_values")
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if kab_values:
            members = [str(kab_values[i]) if kab_values[i] is not None else f"Baris {i+1}" for i in idxs]
        else:
            members = [f"Baris {i+1}" for i in idxs]
        kab_lists.append(members)

    stats = []
    for c in range(k):
        sub = imputed[labels == c]
        stats.append({
            "cluster": c + 1,
            "mean": {h: round(float(sub[h].mean()), 2) for h in headers},
            "std": {h: round(float(sub[h].std(ddof=0)), 2) for h in headers},
            "min": {h: round(float(sub[h].min()), 2) for h in headers},
            "max": {h: round(float(sub[h].max()), 2) for h in headers},
        })

    # Persentase Pengurangan & Penanganan dihitung dari CENTROID (rata-rata tiap
    # klaster), relatif terhadap Timbulan — mengikuti logika skripsi:
    #   % Pengurangan = mean(pengurangan_sampah) / mean(timbulan_tahunan) * 100
    #   % Penanganan  = mean(penanganan_sampah)  / mean(timbulan_tahunan) * 100
    # Kolom dilacak sejak pemilihan variabel di Upload Data & diperbarui otomatis
    # setiap kali user mengganti nama variabelnya (lihat upload_proceed & processing_rename),
    # sehingga tetap valid berapa pun user me-rename variabelnya.
    timbulan_col = state.get("timbulan_col")
    pengurangan_col = state.get("pengurangan_col")
    penanganan_col = state.get("penanganan_col")
    pct_ready = bool(timbulan_col and pengurangan_col and penanganan_col
                      and timbulan_col in headers and pengurangan_col in headers
                      and penanganan_col in headers)

    avg_p, avg_t = [], []
    if pct_ready:
        for c in range(k):
            sub = imputed[labels == c]
            timb_mean = float(sub[timbulan_col].mean())
            peng_mean = float(sub[pengurangan_col].mean())
            tang_mean = float(sub[penanganan_col].mean())
            avg_p.append(round(peng_mean / timb_mean * 100, 2) if timb_mean else 0.0)
            avg_t.append(round(tang_mean / timb_mean * 100, 2) if timb_mean else 0.0)
    else:
        avg_p = [0.0] * k
        avg_t = [0.0] * k

    combined = [(avg_p[c] + avg_t[c]) / 2 for c in range(k)]
    max_v, min_v = max(combined), min(combined)

    conclusions = []
    for c in range(k):
        if k <= 2:
            level = "TINGGI" if combined[c] == max_v else "RENDAH"
        else:
            rng = (max_v - min_v) or 1
            rel = (combined[c] - min_v) / rng
            level = "TINGGI" if rel > 0.66 else ("RENDAH" if rel < 0.34 else "SEDANG")
        conclusions.append({
            "cluster": c + 1, "avg_p": avg_p[c], "avg_t": avg_t[c], "level": level,
            "color": CLUSTER_COLORS[c],
        })

    sil = state["final_silhouette"]
    dbi = state["final_dbi"]
    sil_label, _ = silhouette_interp(sil)
    dbi_label, dbi_detail = dbi_interp(dbi)

    return render_template(
        "interpretation.html",
        unlocked=unlocked_map(state),
        k=k, counts=counts, pie_labels=pie_labels, n=n,
        colors=CLUSTER_COLORS,
        prov_table=prov_table, kab_lists=kab_lists,
        stats=stats, headers=headers,
        pct_ready=pct_ready, timbulan_col=timbulan_col,
        pengurangan_col=pengurangan_col, penanganan_col=penanganan_col,
        avg_p=avg_p, avg_t=avg_t, conclusions=conclusions,
        sil=sil, sil_label=sil_label, dbi=dbi, dbi_label=dbi_label, dbi_detail=dbi_detail,
    )


# ====================================================================
@app.route("/reset", methods=["POST"])
def reset_all():
    sid = session.get("sid")
    if sid and sid in STORE:
        STORE.pop(sid)
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)