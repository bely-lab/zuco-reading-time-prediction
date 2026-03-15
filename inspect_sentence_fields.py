import h5py

FILE_PATH = "data/resultsYAC_NR.mat"

def deref(f, ref):
    """Dereference a MATLAB v7.3 object reference."""
    return f[ref]

def read_matlab_string(f, ref):
    """Read MATLAB string stored as object reference."""
    obj = deref(f, ref)
    # MATLAB strings often stored as uint16 codes
    data = obj[()]
    try:
        return "".join(chr(c) for c in data.flatten())
    except Exception:
        return str(data)

with h5py.File(FILE_PATH, "r") as f:
    sd = f["sentenceData"]

    # pick sentence 0
    content_ref = sd["content"][0, 0]
    sent_text = read_matlab_string(f, content_ref)
    print("Sentence example:\n", sent_text[:200], "...\n")

    word_ref = sd["word"][0, 0]
    word_group = deref(f, word_ref)

    print("Keys inside sentenceData.word (for one sentence):")
    for k in word_group.keys():
        print(" -", k)

    # Check TRT existence
    if "TRT" in word_group.keys():
        trt_ref = word_group["TRT"][0, 0]
        trt_obj = deref(f, trt_ref)
        print("\nExample TRT raw object:", trt_obj[()])