# tlg-avoid

`tlg-avoid` contains code to reproduce a time-geographic analysis of tiger–leopard interactions and a correlated random walk (CRW) null model for testing **intentional** (attraction/avoidance) versus **incidental** (random encounter) co-occurrence from GPS tracking data.

The workflow has two main components:

- **CRW null model**: Monte Carlo simulations constrained by movement distributions (step lengths and turning angles) and environmental probability surfaces (e.g., slope or prey layers), with a validity check to keep simulated steps within the accessible landscape.
- **Time-geographic interaction analysis**: ORTEGA-based quantification of concurrent and lagged interactions across multiple time-delay bins, enabling comparison of observed interaction regimes to null expectations.

---

## Citation info

**Under review.**  
A manuscript describing the method and case study is currently under review. A formal citation will be added after acceptance/publication.

---

## Repository structure

```         
tlg-avoid
│   README.md
│   LICENSE
└───Code
    │   1. Tiger_leopard_CRW_NullModel.ipynb
    │   2. Visualization.ipynb
    │   workers_null_model.py
```
---

## Data
This repository does **not** include the full GPS tracking dataset or environmental rasters due to the sensitive nature of carnivore locations.

To run the notebooks, prepare the following inputs (or adjust the code to match your data schema):

### 1) GPS tracking data (tabular)
Minimum required fields:
- individual ID (e.g., `idcollar`)
- timestamp (e.g., `Time_LMT`)
- projected coordinates (e.g., `proj_lon`, `proj_lat` in UTM)
- step length and turning angle (or enough information to derive them)

### 2) Environmental rasters (GeoTIFF)
At least one raster layer used to build an environmental probability surface:
- `slope`

Optionally include prey layers such as `sb`, `bt`, `gr`.

Ensure rasters align with the tracking coordinate system (same CRS and units) or reproject before analysis.

---

## Getting Started

1. **Clone the repository**: Clone or download this repository to your local computer.
2. **Data setup**: Prepare the tracking dataset and at least one raster layer.
3. **Run the null model + interaction analysis**: Open `Code/1. Tiger_leopard_CRW_NullModel.ipynb` and run the workflow.  `Code/workers_null_model.py` contains helper functions and can be used for parallel processing.
4. **Visualize and summarize results**: Open `Code/2. Visualization.ipynb` to read outputs from the simulation and interaction analysis steps and generate summary tables and figures.

---

## License

`tlg-avoid` is licensed under the MIT license. See `LICENSE` for details.
