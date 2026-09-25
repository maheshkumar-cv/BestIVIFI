# Parametrized interval-valued intuitionistic fuzzy set for low-light image enhancement

Python implementation and interactive demo.
The method enhances low-light images by:

1. Converting the image from RGB to **HSI** (Hue, Saturation, Intensity) space.
2. Jointly searching for the **(γ, α)** pair that maximizes entropy of the
   intensity channel `I`, where γ and α parameterizes interval-valued intuitionistic fuzzy.
3. Applying **CLAHE** to the resulting intensity image (once γ and α are
   finalized — CLAHE is not part of the entropy search itself).
4. Converting back from HSI to RGB.

Only the intensity channel is modified — hue and saturation are carried
through unchanged. 

**Live demo:** https://bestivifi.streamlit.app

## Paper

> **Parametrized interval valued intuitionistic fuzzy set for low light image
> enhancement**
> Maheshkumar, C.V., David Raj, M., Saraswathi, D.
> *2025 12th International Conference on Soft Computing & Machine
> Intelligence (ISCMI)*, pages 239–243, 2025

See [Citation](#citation) below for the full BibTeX entry.

## Repository contents

| File | Description |
|---|---|
| `ivifi_enhance.py` | Self-contained pipeline: `rgb2hsi`/`hsi2rgb`/`entropy`/`clahe`, the joint (γ, α) entropy-optimization search, and seven non-membership generators (`mahesh_tan`, `maheshgamma`, `sugeno`, `ragav`, `ravi`, `chithra`, `haribabu`). Also runnable as a script. |
| `app_ivifi.py` | Interactive Streamlit demo that visualizes every stage of the pipeline — H, S, I channels, the (γ, α) search loop as an entropy heatmap, the enhanced (CLAHE'd) intensity, and the final image. |
| `requirements.txt` | Python dependencies. |
| `sample.png` | *(optional)* Bundle your own low-light test image under this name and both scripts will use it as their default input. |

## Installation

```bash
git clone https://github.com/mahesh0961/BestIVIFI.git
cd BestIVIFI
pip install -r requirements.txt
```


## Usage

### Command line

```bash
# Uses sample_image.jpg in this folder, mahesh_tan generator, by default
python ivifi_enhance.py

# Or specify input, output, and generator explicitly
python ivifi_enhance.py path/to/your_image.jpg path/to/output.png sugeno
```

Prints the optimal γ, optimal α, and the number of outer iterations the
search took.

> **Note on runtime:** `sugeno`, `ragav`, `ravi`, `chithra`, and `haribabu`
> search a 1000×10 (γ, α) grid on their first iteration — this can take
> several seconds to tens of seconds depending on image size. `mahesh_tan`
> (15 points) and `maheshgamma` (10 points) are fast.

### Interactive demo

```bash
streamlit run app.py
```

Opens a browser tab where you can:
- Upload your own image (or use the bundled `sample.png` / a stock
  fallback image if none is present).
- Choose among the seven generator methods.
- Inspect the RGB → H, S, I decomposition.
- Step through every iteration of the (γ, α) search, viewing it as an
  entropy heatmap with the winning cell marked.
- Compare the original and enhanced intensity channels with linear- and
  log-scale histograms.
- View and download the final enhanced RGB image.

A hosted version of this demo is live at **https://bestivifi.streamlit.app** — no
installation needed.


## Using the library in your own code

```python
import ivifi_enhance as ivife
from skimage import io

A = io.imread("your_image.jpg")
hsi = ivife.rgb2hsi(A)
H, S, I = hsi[..., 0], hsi[..., 1], hsi[..., 2]

# IF_image already has CLAHE applied (pass apply_clahe=False for the raw result)
IF_image, gamma, alpha, iterations = ivife.optimal_interval(I, "mahesh_tan")

enhanced_hsi = hsi.copy()
enhanced_hsi[..., 2] = IF_image
enhanced_rgb = ivife.hsi2rgb(enhanced_hsi)
```

Or use the convenience wrapper that does the full RGB→RGB round trip:

```python
enhanced_rgb, gamma, alpha, iterations = ivife.pro_ivifi(A, "sugeno")
```

Available generator names: `"mahesh_tan"`, `"maheshgamma"`, `"sugeno"`,
`"ragav"`, `"ravi"`, `"chithra"`, `"haribabu"`.

## Citation

If you use this code, please cite the paper:

```bibtex
@INPROCEEDINGS{2025Mahesha,
	author = {Maheshkumar, C.V. and David Raj, M. and Saraswathi, D.},
	booktitle={2025 12th International Conference on Soft Computing \& Machine Intelligence (ISCMI)}, 
	title={Parametrized interval valued intuitionistic fuzzy set for low light image enhancement}, 
	year={2025},
	volume={},
	number={},
	pages={239-243},
	keywords={Image enhancement; interval valued intuitionistic fuzzy set; Searching algorithm; Parametrization; CLAHE}}
```

## Contact

For questions about the code or the paper, feel free to email
maheshkumarcv961@gmail.com.
