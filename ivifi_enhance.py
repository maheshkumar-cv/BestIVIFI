
import sys
from pathlib import Path

import numpy as np
from skimage import io, exposure, img_as_float

EPS = np.finfo(float).eps
DEFAULT_IMAGE = Path(__file__).with_name("sample.png")


def rgb2hsi(rgb):
    rgb = img_as_float(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(num / (den + EPS), -1.0, 1.0))

    H = theta.copy()
    H[b > g] = 2 * np.pi - H[b > g]
    H = H / (2 * np.pi)

    S = 1 - 3 * np.minimum(np.minimum(r, g), b) / (r + g + b + EPS)
    I = (r + g + b) / 3.0
    return np.dstack([H, S, I])


def hsi2rgb(hsi):
    h, s, i = (hsi[..., k].astype(float) for k in range(3))
    theta = h * 2 * np.pi

    achro = s == 0
    m1 = ~achro & (theta < 2 * np.pi / 3)
    m2 = ~achro & (theta >= 2 * np.pi / 3) & (theta < 4 * np.pi / 3)
    m3 = ~achro & (theta >= 4 * np.pi / 3)

    t = theta.copy()
    t[m2] -= 2 * np.pi / 3
    t[m3] -= 4 * np.pi / 3

    with np.errstate(divide="ignore", invalid="ignore"):
        lo = i * (1 - s)
        f = i * (1 + (s * np.cos(t)) / np.cos(np.pi / 3 - t))
        hi = 3 * i - (f + lo)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    r[achro] = g[achro] = b[achro] = i[achro]
    # sector 1: 0 <= theta < 120 deg
    b[m1], r[m1], g[m1] = lo[m1], f[m1], hi[m1]
    # sector 2: 120 <= theta < 240 deg
    r[m2], g[m2], b[m2] = lo[m2], f[m2], hi[m2]
    # sector 3: 240 <= theta < 360 deg
    g[m3], b[m3], r[m3] = lo[m3], f[m3], hi[m3]

    out = np.dstack([r, g, b]) * 255.0
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)



def entropy(img):
    u = np.rint(np.clip(np.nan_to_num(img, nan=0.0), 0, 1) * 255).astype(np.uint8)
    counts = np.bincount(u.ravel(), minlength=256)
    p = counts[counts > 0] / u.size
    return float(-np.sum(p * np.log2(p)))


def clahe(I, clip_limit=0.01, nbins=256):
    """Stand-in for MATLAB adapthisteq (uniform distribution, ~8x8 tiles)."""
    return exposure.equalize_adapthist(np.clip(I, 0, 1), clip_limit=clip_limit, nbins=nbins)



def _tan(I, g):
    return (1 - I) / (1 + I * np.tan(g))


def _ragav(I, g):
    return (1 - I) / (1 + ((g + 1) ** 3) * I)


def _ravi(I, g):
    return (1 - I) / (1 + 3 * (g ** 2 + 4 * g + 1) * I)


def _haribabu(I, g):
    return (1 - I) / (1 + g * np.exp(g + 1) * I)


def _sugeno(I, g):
    return (1 - I) / (1 + g * I)


def _chithra(I, g):
    return (1 - I) / (1 + ((g + 1) ** 2) * I)


def _mahesh_gamma(I, g):
    return (1 - I) / (1 + I * ((1 - g) / g))



_METHODS = {
    "mahesh_tan":   (_tan,          (0, 1.5), 10),
    "ragav":        (_ragav,        (0, 10),  100),
    "ravi":         (_ravi,         (0, 10),  100),
    "haribabu":     (_haribabu,     (0, 10),  100),
    "sugeno":       (_sugeno,       (0, 10),  100),
    "chithra":      (_chithra,      (0, 10),  100),
    "maheshgamma":  (_mahesh_gamma, (0, 1),   10),
}


def get_transform(name):
    """Generator f(I, gamma) for a given method name."""
    return _METHODS[name][0]


def _entropy_row(I, generator, gamma_val, alpha):
    """Entropy of I + alpha[j] * Hesitation(I, gamma_val), for every alpha[j]."""
    non_m = generator(I, gamma_val)
    hesitation = 1 - I - non_m
    ent = np.empty(alpha.shape[0])
    for j, a in enumerate(alpha):
        ent[j] = entropy(I + a * hesitation)
    return ent


def optimal_interval(I, name, history=None, apply_clahe=True, clip_limit=0.01, nbins=256):
    """Jointly search (gamma, alpha) to maximise entropy of I + alpha*Hesitation,
    then apply CLAHE to the resulting intensity image.

    Returns (IF_image, final_gamma, final_alpha, loop) where `loop` is the
    number of outer iterations, and IF_image is the CLAHE-enhanced intensity
    channel (pass apply_clahe=False to get the raw, pre-CLAHE result instead).
    The entropy search itself always runs on the raw I + alpha*Hesitation
    image -- CLAHE is applied once, after (gamma, alpha) are finalised, not
    inside the search loop.

    If `history` is a list, one dict per outer iteration is appended with
    the gamma/alpha grids, the full entropy matrix, the winning cell, k, and
    the stopping value.
    """
    generator, interval, points_per_unit = _METHODS[name]
    epsilon = 1e-4

    lower_limit = interval[0] + 0.1
    upper_limit = interval[1]
    no_gamma = int((interval[1] - interval[0]) * points_per_unit)
    alpha_ll, alpha_ul = 0.1, 1.0

    previous_E = 0.0
    value = 1.0
    k = 1
    loop = 0
    final_gamma = final_alpha = None

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        while True:
            loop += 1
            gamma = np.linspace(lower_limit, upper_limit, no_gamma)
            alpha = np.linspace(alpha_ll, alpha_ul, 10)

            ent = np.empty((no_gamma, 10))
            for i, g in enumerate(gamma):
                ent[i, :] = _entropy_row(I, generator, g, alpha)

            E = float(np.nanmax(ent))

            if history is not None:
                history.append(dict(
                    iteration=loop,
                    gamma=gamma.copy(),
                    alpha=alpha.copy(),
                    entropy=ent.copy(),
                    best_entropy=E,
                    k=k,
                    value=float(value),
                    converged=bool(value <= epsilon),
                ))

            if value <= epsilon:
                break

            i, j = np.unravel_index(np.nanargmax(ent), ent.shape)
            final_gamma = float(gamma[i])
            final_alpha = float(alpha[j])

            interior_gamma = i not in (0, no_gamma - 1)
            interior_alpha = j not in (0, 9)

            if interior_gamma and interior_alpha:
                k += 1
                value = E - previous_E
                previous_E = E
                lower_limit = gamma[i] - 4 / 10.0 ** k
                upper_limit = gamma[i] + 5 / 10.0 ** k
                alpha_ll = alpha[j] - 4 / 10.0 ** k
                alpha_ul = alpha[j] + 5 / 10.0 ** k
            elif interior_gamma and not interior_alpha:
                value = E - previous_E
                previous_E = E
                alpha_ll = alpha[j] - 4 / 10.0 ** k
                alpha_ul = alpha[j] + 5 / 10.0 ** k
                # gamma range unchanged; only the point count shrinks below
            elif not interior_gamma and interior_alpha:
                value = E - previous_E
                previous_E = E
                lower_limit = gamma[i] - 4 / 10.0 ** k
                upper_limit = gamma[i] + 5 / 10.0 ** k
                # alpha range unchanged
            else:
                value = 1.0
                lower_limit = gamma[i] - 4 / 10.0 ** k
                upper_limit = gamma[i] + 5 / 10.0 ** k
                alpha_ll = alpha[j] - 4 / 10.0 ** k
                alpha_ul = alpha[j] + 5 / 10.0 ** k

            no_gamma = 10

    non_m = generator(I, final_gamma)
    hesitation = 1 - I - non_m
    IF_image = I + final_alpha * hesitation
    if apply_clahe:
        IF_image = clahe(IF_image, clip_limit=clip_limit, nbins=nbins)
    return IF_image, final_gamma, final_alpha, loop


def pro_ivifi(A, name, history=None, apply_clahe=True, clip_limit=0.01, nbins=256):
    """RGB -> HSI -> optimal_interval + CLAHE on I -> HSI -> RGB
    (H, S carried through unchanged)."""
    hsi = rgb2hsi(A)
    H, S, I = hsi[..., 0], hsi[..., 1], hsi[..., 2]
    IF_image, gamma, alpha, loop = optimal_interval(
        I, name, history=history, apply_clahe=apply_clahe,
        clip_limit=clip_limit, nbins=nbins,
    )
    out = hsi.copy()
    out[..., 2] = IF_image
    return hsi2rgb(out), gamma, alpha, loop


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
if __name__ == "__main__":
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("output_interval.png")
    method = sys.argv[3] if len(sys.argv) > 3 else "mahesh_tan"

    if not in_path.exists():
        sys.exit(
            f"Image not found: {in_path}\n"
            f"Usage: python {Path(__file__).name} [input_image] [output_image] [method]\n"
            f"Methods: {', '.join(_METHODS)}"
        )

    A = io.imread(in_path)
    if A.ndim == 3 and A.shape[2] == 4:
        A = A[..., :3]

    output, gamma, alpha, loop = pro_ivifi(A, method)

    print(f"input        = {in_path}")
    print(f"method       = {method}")
    print(f"final gamma  = {gamma:.6f}")
    print(f"final alpha  = {alpha:.6f}")
    print(f"iterations   = {loop}")
    io.imsave(out_path, output)
    print(f"saved        = {out_path}")