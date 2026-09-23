
import inspect
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage import data as skdata
from skimage import img_as_ubyte
from skimage import io as skio
from skimage import transform

import ivifi_enhance as ivife

st.set_page_config(page_title="IVIFI · interval-valued fuzzy enhancement", layout="wide")

if "history" not in inspect.signature(ivife.optimal_interval).parameters or not hasattr(ivife, "clahe"):
    st.error(
        "The `ivifi_enhance.py` being imported is an older version (no `history` argument / "
        "`clahe()`).\n\n"
        f"Loaded from: `{ivife.__file__}`\n\n"
        "Replace it with the latest `ivifi_enhance.py`, then stop Streamlit (Ctrl+C) and run "
        "`streamlit run app_ivifi.py` again so the module is re-imported."
    )
    st.stop()

METHODS = {
    "mahesh_tan":  "tan(γ)  ·  15-pt grid",
    "maheshgamma": "(1-γ)/γ  ·  10-pt grid",
    "sugeno":      "γ  ·  1000-pt grid",
    "ragav":       "(γ+1)³  ·  1000-pt grid",
    "ravi":        "3(γ²+4γ+1)  ·  1000-pt grid",
    "chithra":     "(γ+1)²  ·  1000-pt grid",
    "haribabu":    "γ·e^(γ+1)  ·  1000-pt grid",
}
HEAVY_METHODS = {"sugeno", "ragav", "ravi", "chithra", "haribabu"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def to_u8(x):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def show(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except (TypeError, ValueError):
        st.image(img, caption=caption, use_container_width=True)


DEFAULT_IMAGE = Path(__file__).with_name("sample_image.jpg")


def load_image(file_bytes, max_side):
    if file_bytes is None:
        if DEFAULT_IMAGE.exists():
            A = skio.imread(DEFAULT_IMAGE)
            if A.ndim == 2:
                A = np.dstack([A] * 3)
            A = img_as_ubyte(A[..., :3])
        else:
            try:
                A = skdata.astronaut()
                A = to_u8((A / 255.0) ** 2.2)          # darken -> low-light demo image
            except Exception:
                yy, xx = np.mgrid[0:256, 0:256]
                A = np.dstack([xx, yy, (xx + yy) / 2]).astype(float) / 255.0
                A = to_u8(A ** 2.5)
    else:
        A = skio.imread(io.BytesIO(file_bytes))
        if A.ndim == 2:
            A = np.dstack([A] * 3)
        A = img_as_ubyte(A[..., :3])
    h, w = A.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        A = img_as_ubyte(transform.resize(A, (int(h * s), int(w * s)), anti_aliasing=True))
    return A


@st.cache_data(show_spinner=False)
def run_pipeline(A, method):
    hsi = ivife.rgb2hsi(A)
    H, S, I = hsi[..., 0], hsi[..., 1], hsi[..., 2]

    history = []
    IF_image, gamma, alpha, loop = ivife.optimal_interval(I, method, history=history)

    B = hsi.copy()
    B[..., 2] = IF_image
    out = ivife.hsi2rgb(B)
    return dict(H=H, S=S, I=I, IF=IF_image, gamma=gamma, alpha=alpha,
                loop=loop, history=history, out=out)


def hist_fig(arrs, labels, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    x = np.linspace(0, 1, 256)
    for a, lab in zip(arrs, labels):
        h, _ = np.histogram(np.clip(a, 0, 1), bins=256, range=(0, 1))
        frac = h / h.sum()
        axes[0].plot(x, frac, label=lab, lw=1.2)
        axes[1].plot(x, np.where(h > 0, frac, np.nan), label=lab, lw=1.2)
    axes[1].set_yscale("log")
    axes[0].set_title(f"{title} – linear", fontsize=10)
    axes[1].set_title(f"{title} – log scale", fontsize=10)
    axes[0].set_ylabel("fraction of pixels")
    axes[1].set_ylabel("fraction of pixels (log)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlabel("intensity")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
st.title("Parametrized interval-valued intuitionistic fuzzy enhancement")
st.caption("RGB → HSI → jointly optimise (γ, α) on I (entropy search) → HSI → RGB")

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    max_side = st.slider("Max image side (px)", 64, 1024, 192, 32,
                          help="The joint γ/α search can test up to 10,000 (γ, α) pairs on the "
                               "first iteration for the heavier methods — larger images make "
                               "that noticeably slower.")
    st.header("Method")
    method = st.selectbox("Generator", list(METHODS), format_func=METHODS.get)
    if method in HEAVY_METHODS:
        st.warning(
            "This method searches a 1000×10 grid on its first iteration. "
            "It can take several seconds to tens of seconds depending on image size."
        )

A = load_image(up.getvalue() if up else None, max_side)
if up is None:
    src = "the bundled sample image" if DEFAULT_IMAGE.exists() else "a darkened stock image"
    st.info(f"No file uploaded – showing {src}. Upload your own in the sidebar.")

with st.spinner("Running the joint γ/α search…"):
    R = run_pipeline(A, method)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Outer iterations", R["loop"])
c2.metric("Optimal γ", f"{R['gamma']:.6f}")
c3.metric("Optimal α", f"{R['alpha']:.6f}")
c4.metric("Entropy: I → IF+CLAHE", f"{ivife.entropy(R['I']):.3f} → {ivife.entropy(R['IF']):.3f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "1 · RGB → H, S, I",
    "2 · (γ, α) search loop",
    "3 · Enhanced intensity",
    "4 · Final RGB",
])

# ----------------------------------------------------------------------------
# tab 1 – HSI split
# ----------------------------------------------------------------------------
with tab1:
    cols = st.columns(4)
    items = [("Original RGB", A), ("H – hue", to_u8(R["H"])),
             ("S – saturation", to_u8(R["S"])), ("I – intensity", to_u8(R["I"]))]
    for col, (name, img) in zip(cols, items):
        with col:
            show(img, name)
    st.write("")
    stats = pd.DataFrame({
        "channel": ["H", "S", "I"],
        "min": [R[k].min() for k in "HSI"],
        "max": [R[k].max() for k in "HSI"],
        "mean": [R[k].mean() for k in "HSI"],
        "entropy": [ivife.entropy(R[k]) for k in "HSI"],
    })
    st.dataframe(stats, hide_index=True)
    st.caption("Only I is jointly optimised over (γ, α); H and S are carried through unchanged.")

# ----------------------------------------------------------------------------
# tab 2 – the loop
# ----------------------------------------------------------------------------
with tab2:
    hist = R["history"]
    st.subheader("Convergence across outer iterations")
    summary = pd.DataFrame([{
        "iteration": h["iteration"],
        "γ range": f"[{h['gamma'][0]:.4f}, {h['gamma'][-1]:.4f}]  ({len(h['gamma'])} pts)",
        "α range": f"[{h['alpha'][0]:.4f}, {h['alpha'][-1]:.4f}]  (10 pts)",
        "best E": h["best_entropy"],
        "k": h["k"],
        "value (ΔE test)": h["value"],
        "stop?": "✔" if h["converged"] else "",
    } for h in hist])
    st.dataframe(summary, hide_index=True)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    its = [h["iteration"] for h in hist]
    ax.plot(its, [h["best_entropy"] for h in hist], "o-")
    ax.set_title("best entropy per iteration", fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_xticks(its)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Inspect one iteration's (γ, α) entropy grid")
    if len(hist) > 1:
        it = st.slider("Iteration", 1, len(hist), 1)
    else:
        it = 1
        st.caption("Only one iteration was needed.")
    h = hist[it - 1]
    ent = h["entropy"]
    bi, bj = np.unravel_index(np.nanargmax(ent), ent.shape)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"**Iteration {h['iteration']}** – grid of **{ent.shape[0]}×{ent.shape[1]}** "
            f"(γ, α) pairs  \n"
            f"γ range: [{h['gamma'][0]:.5f}, {h['gamma'][-1]:.5f}]  \n"
            f"α range: [{h['alpha'][0]:.5f}, {h['alpha'][-1]:.5f}]  \n"
            f"Winner: γ = **{h['gamma'][bi]:.6f}**, α = **{h['alpha'][bj]:.6f}**, "
            f"E = **{h['best_entropy']:.5f}**  \n"
            f"k = {h['k']}, stopping value = {h['value']:.6f}"
            + ("  \n✅ stopping criterion met → loop ends" if h["converged"]
               else "  \n➡ new grid built around the winner on each axis where it isn't "
                    "at the edge of the current grid")
        )
    with right:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        im = ax.imshow(
            ent.T, origin="lower", aspect="auto", cmap="viridis",
            extent=[h["gamma"][0], h["gamma"][-1], h["alpha"][0], h["alpha"][-1]],
        )
        ax.plot(h["gamma"][bi], h["alpha"][bj], "r*", ms=16, mec="white", mew=0.7)
        ax.set_xlabel("γ")
        ax.set_ylabel("α")
        fig.colorbar(im, ax=ax, label="entropy")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------
# tab 3 – enhanced intensity
# ----------------------------------------------------------------------------
with tab3:
    cols = st.columns(2)
    steps = [
        ("I – original", R["I"]),
        (f"IF+CLAHE – γ={R['gamma']:.4f}, α={R['alpha']:.4f}", R["IF"]),
    ]
    for col, (name, arr) in zip(cols, steps):
        with col:
            show(to_u8(arr), f"{name}  (E={ivife.entropy(arr):.4f})")
    fig = hist_fig([R["I"], R["IF"]], ["I", "IF+CLAHE (interval-optimised)"], "Intensity histograms")
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# tab 4 – final
# ----------------------------------------------------------------------------
with tab4:
    a, b = st.columns(2)
    with a:
        show(A, "Original")
    with b:
        show(R["out"], "Enhanced (H, S unchanged · IF+CLAHE)")
    buf = io.BytesIO()
    Image.fromarray(R["out"]).save(buf, format="PNG")
    st.download_button("Download enhanced PNG", buf.getvalue(), "enhanced_ivifi.png", "image/png")