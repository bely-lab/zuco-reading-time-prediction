from pathlib import Path
import numpy as np
import pandas as pd
import h5py

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_TO_MS = 2.0  # 1 sample = 2 ms in ZuCo

def _as_ref(x):
    try:
        return x.item()
    except Exception:
        return x

def deref_any(f, maybe_ref):
    ref = _as_ref(maybe_ref)
    obj = f[ref]

    # Sometimes MATLAB stores a ref to a dataset that contains a ref
    if isinstance(obj, h5py.Dataset) and obj.dtype == object:
        inner = obj[0, 0]
        return f[_as_ref(inner)]
    return obj

def read_matlab_string(f, ref):
    obj = deref_any(f, ref)
    arr = obj[()]
    if hasattr(arr, "dtype") and arr.dtype.kind in {"u", "i"}:
        try:
            return "".join(chr(int(c)) for c in np.array(arr).flatten())
        except Exception:
            pass
    return str(arr)

def read_scalar(f, ref):
    obj = deref_any(f, ref)
    arr = obj[()]
    if getattr(arr, "size", 0) == 0:
        return np.nan
    return float(np.array(arr).flatten()[0])

def extract_subject_sentence_trt(mat_path: Path):
    with h5py.File(mat_path, "r") as f:
        sd = f["sentenceData"]
        n_sent = sd["content"].shape[0]

        texts = []
        sent_trt_ms = np.full(n_sent, np.nan, dtype=float)

        for i in range(n_sent):
            texts.append(read_matlab_string(f, sd["content"][i, 0]))

            word_obj = deref_any(f, sd["word"][i, 0])
            if "TRT" not in word_obj:
                continue

            trt_refs = word_obj["TRT"]
            trt_vals = []

            for j in range(trt_refs.shape[0]):
                trt_val = read_scalar(f, trt_refs[j, 0])
                if not np.isnan(trt_val):
                    trt_vals.append(trt_val)

            if trt_vals:
                sent_trt_ms[i] = float(np.sum(trt_vals)) * SAMPLE_TO_MS

        return texts, sent_trt_ms

def main():
    mat_files = sorted(DATA_DIR.glob("results*_NR.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No results*_NR.mat files found in {DATA_DIR.resolve()}")

    print(f"Found {len(mat_files)} subject files.")

    all_subject_trt = []
    reference_texts = None

    for fp in mat_files:
        print("Reading:", fp.name)
        texts, trt_ms = extract_subject_sentence_trt(fp)

        if reference_texts is None:
            reference_texts = texts
        else:
            if len(texts) != len(reference_texts):
                print("WARNING: sentence count mismatch in", fp.name)

        all_subject_trt.append(trt_ms)

    trt_matrix = np.vstack(all_subject_trt)
    n_subjects, n_sent = trt_matrix.shape

    out_df = pd.DataFrame({
        "sentence_id": np.arange(n_sent),
        "sentence_text": reference_texts,
        "mean_trt_ms": np.nanmean(trt_matrix, axis=0),
        "std_trt_ms": np.nanstd(trt_matrix, axis=0),
        "n_subjects_used": np.sum(~np.isnan(trt_matrix), axis=0),
    })

    out_path = OUT_DIR / "zuco_nr_sentence_trt.csv"
    out_df.to_csv(out_path, index=False)

    print("\nSaved:", out_path.resolve())
    print("Rows:", len(out_df))
    print(out_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()