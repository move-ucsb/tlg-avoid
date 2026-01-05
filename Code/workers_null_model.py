import pandas as pd
import ortega
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import MultiPoint, mapping
import scipy.stats as stats
from rasterio.mask import mask
from rasterio.features import rasterize
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import math, random
from math import radians,cos,sin,asin,sqrt
import gdal



def analyze_step_length_distribution(df, id_pair):
    """
    Analyzes the step length distribution for a given pair of animals.
    Fits multiple probability distributions and selects the best one.

    """
    # Filter dataset for the selected individuals
    df_pair = df[df['idcollar'].isin(id_pair)]

    # Define distributions to test
    distributions = [stats.gamma, stats.lognorm, stats.expon, stats.weibull_min]

    # Store results
    best_fits = {}

    # Plot step length distributions and fit different distributions
    plt.figure(figsize=(12, 6))

    for i, id in enumerate(id_pair):
        step_lengths = df_pair[df_pair['idcollar'] == id]['stepLength'].dropna()

        # Histogram of step lengths
        plt.subplot(1, 2, i + 1)
        plt.hist(step_lengths, bins=30, density=True, alpha=0.6, color='gray', label='Observed')

        # Fit each distribution and compute log-likelihood
        best_fit = None
        best_ll = -np.inf
        best_params = None

        for dist in distributions:
            params = dist.fit(step_lengths)  # Fit the distribution
            ll = np.sum(dist.logpdf(step_lengths, *params))  # Compute log-likelihood

            # Check if this is the best fit
            if ll > best_ll:
                best_ll = ll
                best_fit = dist
                best_params = params

        # Store best fit
        best_fits[id] = {"distribution": best_fit.name, "parameters": best_params}

        # Plot best-fitting distribution
        x = np.linspace(min(step_lengths), max(step_lengths), 100)
        y = best_fit.pdf(x, *best_params)
        plt.plot(x, y, label=f'Best fit: {best_fit.name}', linewidth=2)

        plt.title(f'Step Length Distribution - {id}')
        plt.xlabel('Step Length (km)')
        plt.ylabel('Density')
        plt.legend()

        # Print best-fitting distribution parameters
        print(f"Best fit for {id}: {best_fit.name}")
        print(f"Parameters: {best_params}")

    plt.tight_layout()
    plt.show()

    return best_fits


def fit_mixture_wrapped_normal(angles):
    """
    Fits a mixture of two wrapped normal distributions to capture bimodality.

    Parameters:
    - angles: np.array, observed turning angles

    Returns:
    - best_params: list of fitted wrapped normal parameters
    """
    def wrapped_normal_mixture_neg_log_likelihood(params):
        """Negative log-likelihood function for a mixture of two wrapped normal distributions."""
        w, mu1, sigma1, mu2, sigma2 = params
        w = np.clip(w, 0.01, 0.99)  # Ensure valid mixture weights
        likelihood = (
            w * stats.vonmises.pdf(angles, sigma1, loc=mu1) +
            (1 - w) * stats.vonmises.pdf(angles, sigma2, loc=mu2)
        )
        return -np.sum(np.log(likelihood + 1e-9))  # Avoid log(0)

    # Initial guesses: one component near -π, one near +π
    initial_params = [0.5, -np.pi, 2, np.pi, 2]  # w, mu1, sigma1, mu2, sigma2
    bounds = [(0.01, 0.99), (-np.pi, np.pi), (0.1, 10), (-np.pi, np.pi), (0.1, 10)]

    # Optimize parameters using MLE
    result = minimize(wrapped_normal_mixture_neg_log_likelihood, initial_params, bounds=bounds)
    best_params = result.x

    return best_params



def analyze_turning_angle_distribution(df, id_pair):
    """
    Analyzes the turning angle distribution for a given pair of animals.
    Fits a Mixture of Two Wrapped Normal Distributions.

    Parameters:
    - df: pandas DataFrame containing tracking data with 'turning_angle' column
    - id_pair: list of two idcollar values (e.g., [37821, 229032])

    Returns:
    - Dictionary with the best-fitting model and parameters for each individual.
    """
    plt.figure(figsize=(12, 6))
    best_fits = {}

    for i, id in enumerate(id_pair):
        angles = df[df['idcollar'] == id]['turning_angle'].dropna().values

        # Histogram + KDE for visualization
        plt.subplot(1, 2, i + 1)
        plt.hist(angles, bins=30, density=True, alpha=0.6, color='gray', label='Observed')
        sns.kdeplot(angles, color='black', linewidth=1.5, label='KDE')

        # Compute mean turn angles
        mean_turn_angle = np.degrees(np.mean(angles))
        
        # Fit Mixture of Wrapped Normal Distributions
        wrapped_normal_params = fit_mixture_wrapped_normal(angles)

        # Compute persist_dir (small turns < 10°)
        threshold = np.radians(10)  # Convert 10 degrees to radians
        persist_dir = np.mean(np.abs(angles) < threshold)

        # Compute std_persist_turns from only small turning angles
        persist_turn_angles = angles[np.abs(angles) < threshold]
        std_persist_turns = np.degrees(np.std(persist_turn_angles))

        # Store best fit
        best_fits[id] = {"distribution": "Mixture of Wrapped Normals", 
                         "parameters": wrapped_normal_params,
                         "persist_dir": persist_dir,
                         "std_persist_turns": std_persist_turns,
                         "mean_turn_angle": mean_turn_angle
                        }

        # Plot best-fitting model
        x = np.linspace(-np.pi, np.pi, 100)
        w, mu1, sigma1, mu2, sigma2 = wrapped_normal_params
        y = (
            w * stats.vonmises.pdf(x, sigma1, loc=mu1) +
            (1 - w) * stats.vonmises.pdf(x, sigma2, loc=mu2)
        )

        plt.plot(x, y, label=f'Best fit: Mixture of Wrapped Normals', linewidth=2)
        plt.title(f'Turning Angle Distribution - {id}')
        plt.xlabel('Turning Angle (radians)')
        plt.ylabel('Density')
        plt.legend()

        # Print best-fitting distribution parameters
        print(f"Best fit for {id}: Mixture of Wrapped Normals")
        print(f"Parameters: {wrapped_normal_params}")
        print(f"Persist_dir for {id}: {persist_dir:.3f}, Std_persist_turns: {std_persist_turns:.3f}, Mean Turn Angle: {mean_turn_angle:.4f} degrees")

    plt.tight_layout()
    plt.show()

    return best_fits



def analyze_slope_distribution(df, id_pair):
    """
    Computes and plots the best-fit probability distributions for slope.
    """
    distributions = [stats.chi2, stats.gamma, stats.lognorm]
    best_fits = {}

    for id in id_pair:
        print(f"\n--- Analyzing Slope Distribution for {id} ---")
        best_fits[id] = {}

        values = df[df["idcollar"] == id]["slope"].dropna().values

        if len(values) == 0:
            print(f"Warning: No slope values found for {id}'s home range.")
            continue

        # Fit multiple distributions and select the best
        best_fit = None
        best_ll = -np.inf
        best_params = None

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=30, density=True, alpha=0.6, color='gray', label='Observed')

        for dist in distributions:
            try:
                params = dist.fit(values)
                ll = np.sum(dist.logpdf(values, *params))

                if ll > best_ll:
                    best_ll = ll
                    best_fit = dist
                    best_params = params

            except Exception as e:
                print(f"Could not fit {dist.name} for slope in {id}: {e}")

        # Store best fit
        if best_fit:
            best_fits[id]["slope"] = {"distribution": best_fit.name, "parameters": best_params}

            # Plot best-fitting distribution
            x = np.linspace(min(values), max(values), 100)
            y = best_fit.pdf(x, *best_params)
            plt.plot(x, y, label=f'Best fit: {best_fit.name}', linewidth=2)

        plt.title(f"Slope Distribution - {id}")
        plt.xlabel("Slope Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

        # Print best-fitting distribution parameters
        print(f"Best fit for slope ({id}): {best_fit.name}")
        print(f"Parameters: {best_params}")

    return best_fits


def check_and_fit_prey_distribution(df, id_pair, threshold=0.6):
    """
    Checks the prey occupancy distribution and fits probability distributions for included prey types.

    Parameters:
    - df: DataFrame containing GPS tracking data
    - id_pair: List of individual IDs (e.g., [37821, 229032])
    - threshold: Minimum occupancy value required for inclusion

    Returns:
    - Dictionary with the best-fitting distribution and parameters for each included prey type per individual.
    """
    # Define candidate distributions
    distributions = [stats.gamma, stats.lognorm, stats.beta, stats.weibull_min]

    best_fits = {}

    for id in id_pair:
        print(f"\n--- Checking and Fitting Prey Distribution for {id} ---")
        best_fits[id] = {}

        plt.figure(figsize=(12, 4))

        for i, prey in enumerate(["sb", "bt", "gr"]):  # Include "gr" if relevant
            values = df[df["idcollar"] == id][prey].dropna()

            # Plot histogram
            plt.subplot(1, 3, i + 1)
            plt.hist(values, bins=30, color='gray', alpha=0.7, density=True)
            plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label="Threshold (0.6)")
            plt.xlabel(f"{prey.capitalize()} Occupancy")
            plt.ylabel("Density")
            plt.title(f"{prey.capitalize()} Occupancy Distribution - {id}")
            plt.legend()

            # # Check if majority of values are below threshold
            # below_threshold = np.sum(values < threshold) / len(values)
            # if below_threshold > 0.5:  # If more than 50% of values are below 0.6, exclude this prey
            #     print(f"Excluding {prey} for {id} (majority occupancy < {threshold})")
            #     continue  # Skip fitting the distribution

            # print(f"Including {prey} for {id} (sufficient occupancy ≥ {threshold})")

            # Fit the best distribution
            best_fit = None
            best_ll = -np.inf
            best_params = None

            for dist in distributions:
                try:
                    params = dist.fit(values)
                    ll = np.sum(dist.logpdf(values, *params))

                    if ll > best_ll:
                        best_ll = ll
                        best_fit = dist
                        best_params = params

                except Exception as e:
                    print(f"Could not fit {dist.name} for {prey} in {id}: {e}")

            # Store best fit
            if best_fit:
                best_fits[id][prey] = {"distribution": best_fit.name, "parameters": best_params}

                # Plot best-fitting distribution
                x = np.linspace(min(values), max(values), 100)
                y = best_fit.pdf(x, *best_params)
                plt.plot(x, y, label=f'Best fit: {best_fit.name}', linewidth=2)

        plt.tight_layout()
        plt.show()

    return best_fits


def clip_and_combine_probability(df, raster_paths, output_folder, id_pair, slope_fits, prey_fits, ssf_weights_dict):
    """
    Clips required raster layers to MCP, aligns them, converts to probability, applies weighted sum using individual SSF coefficients, 
    normalizes, saves the final combined raster, and returns the combined probability arrays.

    Parameters:
    - df: DataFrame containing GPS tracking data with 'proj_lon' and 'proj_lat'
    - raster_paths: Dictionary with raster file paths (keys include "slope", and any prey keys like "bt", "gr", etc.)
    - output_folder: Path to save clipped and combined rasters
    - id_pair: List of individual IDs (e.g., [37821, 229032])
    - slope_fits: Dictionary with best-fit slope distributions
    - prey_fits: Dictionary with best-fit prey distributions
    - ssf_weights_dict: Dictionary of SSF weights for each individual

    Returns:
    - combined_probs: Dictionary mapping each individual ID to its combined probability array.
    """ 

    combined_probs = {}
    prob_means = {}
    combined_masks = {}
    
    for id in id_pair:
        print(f"\nProcessing individual {id}...")

        # Get the SSF weights for this individual
        if id not in ssf_weights_dict:
            print(f"⚠ Warning: No SSF weights found for {id}. Skipping...")
            continue
        ssf_weights = ssf_weights_dict[id]

        # Filter only relevant environmental factors (slope or prey layers)
        included_layers = [layer for layer in ssf_weights if layer in raster_paths]

        if not included_layers:
            print(f"⚠ Skipping {id}: No relevant environmental factors (slope or prey layers).")
            continue  # Skip this individual

        # # Get included environmental layers
        # included_layers = ["slope"]  # Slope is always included
        # included_layers += list(prey_fits.get(id, {}).keys())  # Add only included prey

        # Create MCP (Minimum Convex Polygon)
        subset = df[df["idcollar"] == id]
        points = [tuple(xy) for xy in subset[["proj_lon", "proj_lat"]].values]
        mcp = MultiPoint(points).convex_hull
        mcp_gdf = gpd.GeoDataFrame(geometry=[mcp], crs="EPSG:32647")  # UTM Zone 47N

        clipped_rasters = {}

        for layer in included_layers:
            raster_path = raster_paths[layer]

            with rasterio.open(raster_path) as src:
                # Clip the raster to MCP
                clipped_image, clipped_transform = mask(src, mcp_gdf.geometry, crop=True)

                # Ensure correct raster shape
                if len(clipped_image.shape) == 3 and clipped_image.shape[0] == 1:
                    clipped_image = clipped_image[0]  # Remove singleton band dimension

                # Save metadata
                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    "height": clipped_image.shape[0],  # Updated for correct shape
                    "width": clipped_image.shape[1],
                    "transform": clipped_transform
                })

                # Save clipped raster
                clipped_raster_path = f"{output_folder}/{layer}_clipped_MCP_{id}.tif"
                with rasterio.open(clipped_raster_path, "w", **clipped_meta) as dst:
                    dst.write(clipped_image, 1)

                print(f"Clipped {layer} raster (MCP) saved for {id}: {clipped_raster_path}")

                # Store clipped raster for alignment
                clipped_rasters[layer] = clipped_image

        # rasterize the MCP to a hard mask
        H, W = next(iter(clipped_rasters.values())).shape
        hull_mask = rasterize(
            [(mapping(mcp), 1)],
            out_shape=(H, W),
            transform=clipped_transform,
            fill=0,
            all_touched=True,
            dtype="uint8"
        )
        combined_masks[id] = hull_mask

        # Step 1: Find the Smallest Common Shape (Height & Width)
        min_height = min(image.shape[0] for image in clipped_rasters.values())
        min_width = min(image.shape[1] for image in clipped_rasters.values())

        # Step 2: Resize All Rasters to the Smallest Common Shape
        aligned_rasters = {layer: image[:min_height, :min_width] for layer, image in clipped_rasters.items()}

        # Step 3: Convert Cropped Rasters to Probability using SSF Weights
        combined_prob = np.zeros((min_height, min_width), dtype=np.float32)

        for layer, image in aligned_rasters.items():
            if layer not in ssf_weights:
                continue  # Skip layers without SSF weights

            # # Clip extreme values
            # image = np.clip(image, 0, np.percentile(image, 99))

            # Handle extreme values by clipping within a reasonable range
            if layer == "slope":
                if id in slope_fits and "slope" in slope_fits[id]:  # Ensure slope exists in slope_fits
                    dist_params = slope_fits[id]["slope"]["parameters"]
                    image = np.clip(image, 0, np.percentile(image, 99))  # Clip outliers
                    prob_layer = stats.lognorm.pdf(image, *dist_params)
                else:
                    print(f"⚠ Warning: No slope fit found for {id}, skipping slope layer.")
                    continue
        
            else:  # For prey layers (e.g., bt, gr)
                if id in prey_fits and layer in prey_fits[id]:  # Ensure layer exists in prey_fits
                    dist_params = prey_fits[id][layer]["parameters"]
                    image = np.clip(image, 0, np.percentile(image, 99))  # Clip extreme values
                    prob_layer = stats.beta.pdf(image, *dist_params)
                else:
                    print(f"⚠ Warning: No prey fit found for {id} - {layer}, skipping.")
                    continue

            # Fix: Remove invalid values (NaNs, negative values)
            prob_layer[np.isnan(prob_layer)] = 0
            prob_layer[prob_layer < 0] = 0

            # Fix: Normalize each layer before applying weights
            if np.nanmax(prob_layer) > 0:
                prob_layer = (prob_layer - np.nanmin(prob_layer)) / (np.nanmax(prob_layer) - np.nanmin(prob_layer))
            else:
                prob_layer[:] = 0  # If no valid values, set to zero

            # Fix: Apply log transformation to prevent extreme small values
            prob_layer = np.log1p(prob_layer)

            # Apply weight from SSF
            combined_prob += ssf_weights[layer] * prob_layer

        # Normalize final combined probability raster to 0-1
        if np.nanmax(combined_prob) > 0:
            combined_prob = (combined_prob - np.nanmin(combined_prob)) / (np.nanmax(combined_prob) - np.nanmin(combined_prob))
        else:
            combined_prob[:] = 0  # If no valid values, set to zero

        # Ensure no negative values
        combined_prob[combined_prob < 0] = 0

        # Ensure correct shape
        combined_prob = np.squeeze(combined_prob)
        if combined_prob.ndim == 2:
            combined_prob = combined_prob[np.newaxis, :, :]

        print("Final combined_prob shape:", combined_prob.shape)

        # Compute Mean Probability
        prob_mean = np.nanmean(combined_prob) if np.any(~np.isnan(combined_prob)) else 0
        prob_means[id] = prob_mean
        print(f"Mean probability (prob_mean) for {id}: {prob_mean:.3f}")

        # Save the final raster
        new_meta = clipped_meta.copy()
        new_meta.update({
            "height": combined_prob.shape[1],
            "width": combined_prob.shape[2],
            "count": 1
        })

        combined_output_path = f"{output_folder}/combined_probability_MCP_{id}.tif"
        with rasterio.open(combined_output_path, "w", **new_meta) as dst:
            dst.write(np.nan_to_num(combined_prob, nan=0).astype(rasterio.float32))

        print(f"Final combined probability raster saved for {id}: {combined_output_path}")

        combined_probs[id] = combined_prob

    return combined_probs, prob_means, combined_masks

def clip_and_normalize_slope(
    df_slice,
    raster_paths: Dict[str,str],
    output_folder: Path,
    id_pair: Tuple[int,int],
    slope_fits: Dict[int,Dict]
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, np.ndarray]]:
    """
    For each individual in id_pair:
    - Clip the 'slope' raster to its MCP
    - Compute a lognormal-pdf probability surface
    - Normalize to [0,1]
    - Save both the clipped slope and the final probability raster
    Returns:
      combined_probs[id], prob_means[id], hull_masks[id]
    """
    combined_probs = {}
    prob_means     = {}
    hull_masks     = {}

    output_folder.mkdir(parents=True, exist_ok=True)

    for animal_id in id_pair:
        # 1) Build MCP from projected coords
        sub = df_slice[df_slice["idcollar"] == animal_id]
        mcp = MultiPoint(sub[["proj_lon","proj_lat"]].values).convex_hull
        mcp_gdf = gpd.GeoDataFrame(geometry=[mcp], crs="EPSG:32647")

        # 2) Clip slope raster
        with rasterio.open(raster_paths["slope"]) as src:
            clipped_img, clipped_tf = mask(src, mcp_gdf.geometry, crop=True)
            arr = clipped_img[0] if clipped_img.ndim==3 else clipped_img

            meta = src.meta.copy()
            meta.update({
                "height":    arr.shape[0],
                "width":     arr.shape[1],
                "transform": clipped_tf,
                "count":     1
            })

        # save clipped slope
        clip_path = output_folder/f"slope_clipped_MCP_{animal_id}.tif"
        with rasterio.open(clip_path, "w", **meta) as dst:
            dst.write(arr, 1)

        # 3) Build hull mask
        H, W = arr.shape
        mask_arr = rasterize(
            [(mapping(mcp), 1)],
            out_shape=(H, W),
            transform=clipped_tf,
            fill=0,
            all_touched=True,
            dtype="uint8"
        )
        hull_masks[animal_id] = mask_arr

        # 4) Compute lognormal probability
        params = slope_fits.get(animal_id, {}).get("slope", {}).get("parameters")
        if not params:
            # no fit → zero array
            prob_layer = np.zeros_like(arr, dtype=float)
        else:
            vals = np.clip(arr, 0, np.percentile(arr, 99))
            prob_layer = stats.lognorm.pdf(vals, *params)
            prob_layer[mask_arr==0] = 0

            # normalize to [0,1]
            vmin, vmax = prob_layer.min(), prob_layer.max()
            if vmax > vmin:
                prob_layer = (prob_layer - vmin)/(vmax - vmin)
            else:
                prob_layer[:] = 0

        combined_probs[animal_id] = prob_layer[np.newaxis,:,:]
        prob_means[animal_id]     = float(np.nanmean(prob_layer))

        # 5) Save normalized probability
        prob_path = output_folder/f"combined_probability_MCP_{animal_id}.tif"
        with rasterio.open(prob_path, "w", **meta) as dst:
            dst.write(prob_layer.astype(rasterio.float32), 1)

    return combined_probs, prob_means, hull_masks



# This function reads a raster and converts it to a numpy array
def raster_to_numpy_array(rasterFile):
    # open raster dataset
    dem = gdal.Open(rasterFile)

    print ('Driver: ', dem.GetDriver().ShortName,'/', dem.GetDriver().LongName,)
    print ('Size is ',dem.RasterXSize,'x',dem.RasterYSize,'x',dem.RasterCount)
    print ('Projection is ',dem.GetProjection())
    geotransform = dem.GetGeoTransform()
    if not geotransform is None:
        print ('Origin = (',geotransform[0], ',',geotransform[3],')')
        print ('Pixel Size = (',geotransform[1], ',',geotransform[5],')')
    myArray = np.array(dem.GetRasterBand(1).ReadAsArray())

    return myArray, geotransform

def extract_from_clipped_raster(df, clipped_raster_path, geotransform):
    """
    Extracts environmental values from a clipped raster based on x, y pixel locations.
    Also converts pixel indices to projected coordinates.

    Parameters:
    - df: DataFrame with x, y pixel indices
    - clipped_raster_path: File path to the individual's clipped raster
    - geotransform: Raster geotransform to convert pixel indices to projected coordinates

    Returns:
    - List of extracted values
    - Lists of projected longitude and latitude
    """
    try:
        with rasterio.open(clipped_raster_path) as src:
            raster_data = src.read(1)  # Read the first band
            nodata_val = src.nodata  # NoData value

            extracted_values = []
            proj_lons = []
            proj_lats = []

            for _, row in df.iterrows():
                x_pix, y_pix = int(row["x"]), int(row["y"])  # Convert to integer pixel indices

                if 0 <= x_pix < src.width and 0 <= y_pix < src.height:
                    value = raster_data[y_pix, x_pix]

                    # Handle NoData values
                    if nodata_val is not None and value == nodata_val:
                        value = np.nan

                    extracted_values.append(value)

                    # Convert pixel index to projected coordinates
                    lon = geotransform[0] + x_pix * geotransform[1]
                    lat = geotransform[3] + y_pix * geotransform[5]
                    proj_lons.append(lon)
                    proj_lats.append(lat)
                else:
                    extracted_values.append(np.nan)
                    proj_lons.append(np.nan)
                    proj_lats.append(np.nan)

            return extracted_values, proj_lons, proj_lats
    except Exception as e:
        print(f"⚠ Error reading {clipped_raster_path}: {e}")
        return [np.nan] * len(df), [np.nan] * len(df), [np.nan] * len(df)  # Return NaNs if raster cannot be read


def create_crw_monte_carlo_simulation(mc_stop, num_pts, persis_dir, mu, sigma, sigma_1, 
                                      myArray, cols, rows, envArray, hull_mask, mat_use, time_step, 
                                      start_time, id_selected, best_step_length_fits, best_turning_angle_fits,
                                      mc_start: int = 1):
    
    all_simulated_data = []  # Store all MC simulation trajectories
    # For this example, we run one MC simulation; you can loop over mc_stop if needed.
    for mc_round in range(mc_start, mc_stop + 1):
        print(f"Beginning Monte Carlo iteration {mc_round}/{mc_stop}")
        current_mat_use, xc, yc = crw_combined(num_pts, persis_dir, mu, sigma, sigma_1, 
                                               myArray, cols, rows, envArray, hull_mask, mat_use, time_step,
                                               start_time, id_selected, best_step_length_fits, best_turning_angle_fits)
        # Accumulate visitation matrix across MC runs
        mat_use += current_mat_use

        if len(xc) > 1:
            # total time for the entire track
            total_duration_hours = num_pts * time_step
            # We'll assign times linearly from start_time to start_time + total_duration_hours
            time_list = [
                start_time + pd.Timedelta(hours=(total_duration_hours * i/(len(xc)-1)))
                for i in range(len(xc))
            ]
        else:
            time_list = [start_time] * len(xc)
            
        # Store trajectory with MC round information
        df_mc = pd.DataFrame({
            "idcollar": [id_selected] * len(xc),
            "x": xc,
            "y": yc,
            "time": time_list,
            "MC_round": [mc_round] * len(xc)
        })
        
        all_simulated_data.append(df_mc)

    # Combine all Monte Carlo trajectories into a single DataFrame
    df_all_mc = pd.concat(all_simulated_data, ignore_index=True)
    
    return mat_use, df_all_mc

def crw_combined(num_pts, persis_dir, mu, sigma, sigma_1, 
                 myArray, cols, rows, envArray, hull_mask, mat_use, 
                 time_step, start_time, 
                 id_selected, best_step_length_fits, best_turning_angle_fits,
                 max_tries_per_step = 50
                ):
    """
    Simulated CRW that uses the combined probability raster (envArray) as context.
    """

    print("\nInitializing CRW simulation...")

    # Get movement parameters for the selected individual
    step_params = best_step_length_fits[id_selected]["parameters"]
    turn_params = best_turning_angle_fits[id_selected]["parameters"]
    mix_p, mu1, kappa1, mu2, kappa2 = turn_params
    
    # Choose a random starting point that is valid (within myArray range)
    valid = False
    while not valid:
        new_x = np.random.randint(0, cols)
        new_y = np.random.randint(0, rows)
        if (0 <= new_x < cols and 0 <= new_y < rows and 
            0 <= myArray[new_y, new_x] <= 1 and 
            hull_mask[new_y, new_x] == 1 and
            envArray[new_y, new_x] > 0):
            valid = True
     
    x_coord = [new_x]
    y_coord = [new_y]
    mat_use[new_y, new_x] += 1
    
    # Set initial direction (random 0-360 degrees)
    prev_dir = random.uniform(0, 360)

    print(f"Starting simulation at ({new_x}, {new_y})")
    print("Entering main simulation loop...")

    test_pts = 0  # count of simulated steps
    
    while test_pts <= num_pts:
        test_pts += 1

        if test_pts % 1000 == 0:  # Print every 1000 steps to track progress
            print(f"Step {test_pts}/{num_pts} - Current position: ({new_x}, {new_y})")

        retries = 0
        success = False
        while retries < max_tries_per_step and not success:
            retries += 1
            
            # --- Determine new direction ---
            if random.random() < persis_dir:
                # new_dir = prev_dir + np.random.vonmises(mu2, sigma2)
                new_dir = (prev_dir + np.random.normal(mu, sigma))
            else:
                # new_dir = prev_dir + np.random.vonmises(mu3, sigma3)
                new_dir = (prev_dir + np.random.normal(mu, sigma_1))
            new_dir = test_angle(new_dir)  # adjust to [0,360)
            
            # --- Sample a step length from the fitted lognorm distribution ---
            # Using np.random.lognormal with parameters: mean and sigma of underlying normal.
            step_km = np.random.lognormal(step_params[0], step_params[2]) # increase variation
            step_m  = step_km * 1000
            max_m   = 300 * time_step # 0.3 km/h
            step_m  = min(step_m, max_m)            
    
            # # Limit max step length
            pixel_size = 30
            step_pixels = int(round(step_m / pixel_size))

            # print(f"Step {test_pts}: Step length = {step:.2f}m ")
            
            # Compute endpoint using the step length and new direction (as a vector)
            delta_x, delta_y = compute_angle_distance_move(new_dir, step_pixels)
            t_x = new_x + delta_x
            t_y = new_y + delta_y
        
            # Check if endpoint is valid in DEM (myArray)
            if (t_x < 0 or t_x >= cols or t_y < 0 or t_y >= rows or 
                myArray[t_y, t_x] < 0 or myArray[t_y, t_x] > 1 or  hull_mask[t_y, t_x] == 0 or
                envArray[t_y, t_x] <= 0 or np.isnan(envArray[t_y, t_x])):
                retries += 1
                # print(f"Step {test_pts}: Invalid position ({t_x}, {t_y}), retrying...")
                # If invalid, reinitialize direction randomly and retry this step
                # **Limit retries per step to prevent infinite loops**
                if retries > 100:
                    # print(f"    Too many retries at step {test_pts}, adjusting strategy.")
                    prev_dir = np.random.random() * 360  # Choose a random direction to escape
                    retries = 0  # Reset retry count
                success = False
                continue  # Skip this step

            # success >= accept
            success = True
            new_x, new_y = t_x, t_y
            x_coord.append(new_x)
            y_coord.append(new_y)
            mat_use[new_y, new_x] += 1

            # update direction
            prev_dir = new_dir
            retries = 0

        if not success:
            pass
            
        # End of one step: update current position to last new_x, new_y
        # (They are already updated in the loop.)
    print("CRW Done! number of points =", len(x_coord))
    return mat_use, x_coord, y_coord #, time_list


def test_angle(angle):
    """
    Normalize any angle (in degrees) into the [0, 360) range.
    """
    return angle % 360

def compute_angle_distance_move(new_dir, dist):
    """
    Given a direction (in degrees) and a distance,
    compute the pixel offset (delta_x, delta_y) using a quadrant‐based method.
    Assumes a coordinate system where x increases to the right and y increases downward.
    """
    new_dir = test_angle(new_dir)
    if 0 <= new_dir <= 90:
        delta_x = round(dist * sin(radians(new_dir)))
        delta_y = -round(dist * cos(radians(new_dir)))
    elif 90 < new_dir <= 180:
        delta_x = round(dist * sin(radians(180 - new_dir)))
        delta_y = round(dist * cos(radians(180 - new_dir)))
    elif 180 < new_dir <= 270:
        delta_x = -round(dist * sin(radians(new_dir - 180)))
        delta_y = round(dist * cos(radians(new_dir - 180)))
    elif 270 < new_dir < 360:
        delta_x = -round(dist * sin(radians(360 - new_dir)))
        delta_y = -round(dist * cos(radians(360 - new_dir)))
    else:
        delta_x, delta_y = 0, 0
    return (delta_x, delta_y)

def create_cell_boundary_matrix():
    """
    Creates a point_list representing the eight adjacent cells.
    Two extra points (first two again) are appended to allow wrap-around indexing.
    """
    pl = point_list()
    # Define the eight neighbors (clockwise starting from top-left)
    neighbors = [(-1, -1), (0, -1), (1, -1),
                 (1,  0), (1,  1), (0,  1),
                 (-1, 1), (-1, 0)]
    for pt in neighbors:
        pl.append_point(Point(pt[0], pt[1]))
    # Append the first two points again for easy wrap-around indexing.
    for pt in neighbors[:2]:
        pl.append_point(Point(pt[0], pt[1]))
    return pl


class Point:
    """
    A simple Point class.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def get_X(self):
        return self.x
    def get_Y(self):
        return self.y
    def get_coord(self):
        return (self.x, self.y)

    
class point_list():
    """
    A custom list to hold Point objects and provide convenience methods.
    """
    def __init__(self):
        self.points = []
    def get_length(self):
        return len(self.points)
    def get_Xs(self):
        return [pt.get_X() for pt in self.points]
    def get_Ys(self):
        return [pt.get_Y() for pt in self.points]
    def append_point(self, pt):
        self.points.append(pt)
    def get_point(self, n):
        return self.points[n]
    def plot_point_list_onRaster(self, myArray):
        plt.plot(self.get_Xs(), self.get_Ys(), 'ro-')
        plt.plot(self.get_point(0).get_X(), self.get_point(0).get_Y(), 'gs', label="Start")
        plt.plot(self.get_point(self.get_length()-1).get_X(), self.get_point(self.get_length()-1).get_Y(), 'ys', label="End")
        imgplot = plt.imshow(myArray, vmin=0, vmax=1024)
        imgplot.set_cmap('spectral')
        plt.title("Track on Raster")
        plt.legend()
        plt.show()
    def plot_point_list(self):
        plt.plot(self.get_Xs(), self.get_Ys(), 'ro-')
        plt.plot(self.get_point(0).get_X(), self.get_point(0).get_Y(), 'gs', label="Start")
        plt.plot(self.get_point(self.get_length()-1).get_X(), self.get_point(self.get_length()-1).get_Y(), 'ys', label="End")
        plt.title("Track")
        plt.legend()
        plt.show()


def estimate_time_step_hours(df: pd.DataFrame) -> float:
    """
    Estimate the main time interval (in hours) between GPS points for an individual.
    Uses the median of the time differences to avoid being skewed by noise.
    """
    times = pd.to_datetime(df["time"], errors="coerce").sort_values()
    diffs = times.diff().dropna()
    if diffs.empty:
        return 1.0  # fallback
    return diffs.median().total_seconds() / 3600

        
def process_slice_round(
    slice_label: str,
    mc_round: int,
    lower: float,
    upper: float,
    time_bin_label: str,
    sim_dir: Path,
    id_pair: Tuple[int,int],
    out_dir: Path
) -> Tuple[str,int,str,int,int]:
    """
    1) Read the two simulation parquet files for this slice & MC round
    2) Auto‐derive which attrs we actually used
    3) Run ORTEGA(interaction_analysis)
    4) Dump out events & pairs as parquet
    5) Return (slice_label, mc_round, time_bin_label, n_events, n_pairs)
    """
    
    id1, id2 = id_pair

    # Expected outputs
    ev_path = out_dir / f"{id1}_{id2}_{slice_label}_events_{time_bin_label}_MC{mc_round:04d}.parquet"
    pr_path = out_dir / f"{id1}_{id2}_{slice_label}_pairs_{time_bin_label}_MC{mc_round:04d}.parquet"

    # Skip if already computed
    if ev_path.exists() and pr_path.exists():
        print(f"[{id_pair}],[{slice_label}] MC {mc_round:04d} bin={time_bin_label} already computed")
        return slice_label, mc_round, time_bin_label, 0, 0

    # 1) read the two per‐id slice files
    f1 = sim_dir / f"crw_sim_{id1}_{id2}_{slice_label}_{id1}.parquet"
    f2 = sim_dir / f"crw_sim_{id1}_{id2}_{slice_label}_{id2}.parquet"
    if not f1.exists() or not f2.exists():
        print(f"[{id_pair}],[{slice_label}] ⚠ missing sim parquet(s) for bin={time_bin_label}: "
              f"{'missing ' + str(f1) if not f1.exists() else ''} "
              f"{'missing ' + str(f2) if not f2.exists() else ''}")
        return slice_label, mc_round, time_bin_label, 0, 0
    
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)

    # concat and filter to this MC round
    df_mc = pd.concat([df1, df2], ignore_index=True)
    df_mc = df_mc[df_mc["MC_round"] == mc_round]
    df_mc['time'] = pd.to_datetime(df_mc['time'], errors='coerce').dt.floor('S')

    # Auto‐derive which attrs to hand into ORTEGA
    # run ORTEGA
    orobj = ortega.ORTEGA(
        data             = df_mc,
        latitude_field   = "proj_lat",
        longitude_field  = "proj_lon",
        minute_min_delay = float(lower),
        minute_max_delay = float(upper),
        time_field       = "time",
        id_field         = "idcollar",
        max_el_time_min  = 120.0,
        speed_average    = True,
        attr_fields      = None
    )
    res_obs = orobj.interaction_analysis()
    if res_obs is None or res_obs.df_interaction_events.empty:
        print(f"[{id_pair}],[{slice_label}] ✅ MC {mc_round:04d}, bin={time_bin_label} → 0 events")
        return slice_label, mc_round, time_bin_label, 0, 0

    # maybe compute duration if concurrent bin
    if lower == 0:
        res_obs.compute_interaction_duration()

    # Ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # write events
    df_ev = res_obs.df_interaction_events.copy()
    df_ev["MC_round"] = mc_round
    df_ev["time_bin"]  = time_bin_label
    df_ev.to_parquet(ev_path, index=False)

    # write pairs
    df_pairs = res_obs.df_all_intersection_pairs.copy()
    df_pairs["MC_round"] = mc_round
    df_pairs["time_bin"]  = time_bin_label
    df_pairs.to_parquet(pr_path, index=False)

    print(f"[{id_pair}],[{slice_label}] ✅ MC {mc_round:04d}, bin={time_bin_label} → "
          f"{len(df_ev)} events, {len(df_pairs)} pairs")

    return slice_label, mc_round, time_bin_label, len(df_ev), len(df_pairs)



def process_slice(
    slice_idx: int,
    df_full: pd.DataFrame,
    id_pair: Tuple[int,int],
    raster_paths: dict,
    ssf_weights_dict: dict,
    slice_output_base: Path,
    M_C: int = 10,
    time_bins: List[float] = None,
    time_labels: List[str] = None,
    candidate_attrs: Optional[List[str]] = None
) -> Tuple[int,int,int]:
    """
    1) Cut out slice #slice_idx (1 000 points per animal)
    2) Re‐fit all distributions on that slice
    3) Clip & combine rasters
    4) Run M_C CRW sims → write two per‐ID Parquets
    5) For each (MC_round, time‐bin), call process_slice_round()
    Returns (slice_idx, total_events, total_pairs)
    """
    id1, id2 = id_pair
    slice_label = f"slice{slice_idx:03d}"

    # prepare output folders
    clip_dir = slice_output_base / slice_label / "clipped_rasters"
    sim_dir  = slice_output_base / slice_label / "simulations"
    int_dir  = slice_output_base / slice_label / "interactions"
    for d in (clip_dir, sim_dir, int_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) slice the two tracks
    SLICE = 1000
    df1 = df_full[df_full.idcollar==id1].sort_values("time").reset_index(drop=True)
    df2 = df_full[df_full.idcollar==id2].sort_values("time").reset_index(drop=True)
    n = min(len(df1), len(df2))

    # build your window‐start indices once:
    starts = list(range(0, n, SLICE))
    # if the last window would be too short, anchor it so it ends at n:
    if starts[-1] + SLICE > n and starts[-1] != max(0, n - SLICE):
        starts.append(n - SLICE)
    # now pick the slice indexed by slice_idx
    start = starts[slice_idx]
    end   = start + SLICE
    s1 = df1.iloc[start:end]
    s2 = df2.iloc[start:end]
    df_slice = pd.concat([s1, s2], ignore_index=True)
    
    # if len(s1)<1000 or len(s2)<1000:
    #    return slice_idx, 0, 0    

    df_slice['time'] = pd.to_datetime(df_slice['time'], errors='coerce')\
                       .dt.floor('S')


    # 1b) OBSERVED‐SLICE INTERACTIONS
    # auto‑pick only those SSF layers we actually have in the slice
    valid_attrs = sorted({
        layer
        for iid in id_pair
        for layer in ssf_weights_dict[iid]
        if layer in df_slice.columns
    }) or None

    for lo, hi, lbl in zip(time_bins[:-1], time_bins[1:], time_labels):
        ev_path = int_dir / f"{id1}_{id2}_{slice_label}_events_{lbl}_observed.parquet"
        pr_path = int_dir / f"{id1}_{id2}_{slice_label}_pairs_{lbl}_observed.parquet"
        # **SKIP** if both files already exist
        if ev_path.exists() and pr_path.exists():
            continue
        
        or_obs = ortega.ORTEGA(
            data             = df_slice,
            latitude_field   = "proj_lat",    
            longitude_field  = "proj_lon",    
            minute_min_delay = float(lo),
            minute_max_delay = float(hi),
            time_field       = "time",
            id_field         = "idcollar",
            max_el_time_min  = 120.0,
            speed_average    = True,
            attr_fields      = valid_attrs
        )
        res_obs = or_obs.interaction_analysis()
        if res_obs is None or res_obs.df_interaction_events.empty:
            continue

        # if it’s the concurrent bin, compute durations
        if lo == 0:
            res_obs.compute_interaction_duration()

        # write out events & pairs for observed slice
        ev = res_obs.df_interaction_events.copy()
        ev["slice"]     = slice_idx
        ev["time_bin"]  = lbl
        ev.to_parquet(ev_path, index=False)

        pr = res_obs.df_all_intersection_pairs.copy()
        pr["slice"]    = slice_idx
        pr["time_bin"] = lbl
        pr.to_parquet(pr_path, index=False)

 
    # 2) re‐fit distributions
    slf   = analyze_step_length_distribution(df_slice,   id_pair)
    taf   = analyze_turning_angle_distribution(df_slice, id_pair)
    sf    = analyze_slope_distribution(df_slice,         id_pair)

    # 3) clip & combine rasters
    combined_probs, prob_means, combined_masks = clip_and_normalize_slope(
        df_slice, raster_paths, clip_dir, id_pair, slope_fits=sf)

    # 4) run M_C CRW sims
    for id_sel in id_pair:
        persis = taf[id_sel]["persist_dir"]
        mu      = taf[id_sel]["mean_turn_angle"]
        sigma, sigma_1 = 10, 45

        # load clipped raster
        dem_file = clip_dir / f"combined_probability_MCP_{id_sel}.tif"
        myArray, geotransform = raster_to_numpy_array(str(dem_file))
        envArr = combined_probs[id_sel]
        mask   = combined_masks[id_sel]
        if envArr.ndim==3 and envArr.shape[0]==1:
            envArr = envArr[0]

        df_obs = df_slice[df_slice.idcollar == id_sel].copy()
        df_obs["time"] = pd.to_datetime(df_obs["time"], errors="coerce")

        time_step_hours = estimate_time_step_hours(df_obs)
        time_step_hours = max(0.25, min(time_step_hours, 4.0))  # between 15 mins and 4 hours
        print(f"time_step_hours for {id_sel}: {time_step_hours:.2f}")

        t0 = pd.to_datetime(df_obs.loc[df_obs.index[0], "time"])

        _, df_mc_all = create_crw_monte_carlo_simulation(
            M_C, len(df_obs), persis, mu, sigma, sigma_1,
            myArray, myArray.shape[1], myArray.shape[0],
            envArr, mask, np.zeros_like(myArray,int),
            time_step_hours, t0, id_sel, slf, taf
        )

        # Extract environmental values and projected coordinates
        proj_lons, proj_lats = [], []  # Store projected coordinates
        for layer_name in ["slope", "bt", "gr", "sb"]:
            # Check if the individual uses this layer
            if layer_name not in ssf_weights_dict[id_sel]:
                print(f"Skipping {layer_name} for {id_sel}, not used in SSF.")
                continue
        
            # Define clipped raster path
            clipped_raster_path = f"{clip_dir}/{layer_name}_clipped_MCP_{id_sel}.tif"
        
            # Extract values only if the clipped raster exists
            env_values, proj_lons, proj_lats = extract_from_clipped_raster(df_mc_all, clipped_raster_path, geotransform)
            df_mc_all[layer_name] = env_values  # Assign extracted raster values
        
        # Assign projected coordinates to DataFrame
        df_mc_all["proj_lon"] = proj_lons
        df_mc_all["proj_lat"] = proj_lats

        outp = sim_dir / f"crw_sim_{id1}_{id2}_{slice_label}_{id_sel}.parquet"
        df_mc_all.to_parquet(outp, index=False)

    # 5) now for each MC×bin call process_slice_round()
    total_events = total_pairs = 0
    for MC in range(1, M_C+1):
        for lo, hi, lbl in zip(time_bins, time_bins[1:], time_labels):
            sl, mc, tb, ne, np_ = process_slice_round(
                slice_label, MC, lo, hi, lbl,
                sim_dir, id_pair, int_dir,  ssf_weights_dict
            )
            total_events += ne
            total_pairs  += np_

    return slice_idx, total_events, total_pairs


def process_slice_time(
    slice_idx: int,
    window_start: pd.Timestamp,
    window_end:   pd.Timestamp,
    df_full: pd.DataFrame,
    id_pair: Tuple[int,int],
    raster_paths: dict,
    ssf_weights_dict: dict,
    slice_output_base: Path,
    M_C: int = 10,
    time_bins: List[float] = None,
    time_labels: List[str] = None,
    candidate_attrs: Optional[List[str]] = None
) -> Tuple[int,int,int]:
    """
    1) Cut out slice #slice_idx (1 000 hours window)
    2) Re‐fit all distributions on that slice
    3) Clip & combine rasters
    4) Run M_C CRW sims → write two per‐ID Parquets
    5) For each (MC_round, time‐bin), call process_slice_round()
    Returns (slice_idx, total_events, total_pairs)
    """
    import pandas as pd

    id1, id2 = id_pair
    slice_label = f"slice{slice_idx:03d}"

    # prepare output folders
    clip_dir = slice_output_base / slice_label / "clipped_rasters"
    sim_dir  = slice_output_base / slice_label / "simulations"
    int_dir  = slice_output_base / slice_label / "interactions"
    for d in (clip_dir, sim_dir, int_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────────────
    # 1) SLICE BY A 1 000‑HOUR TIME WINDOW RATHER THAN POINT COUNT
    overall_start = df_full['time'].min()
    overall_end   = df_full['time'].max()
    window        = pd.Timedelta(hours=1000)

    # build list of (start, end) windows
    windows = []
    cur = overall_start
    while cur + window <= overall_end:
        windows.append((cur, cur + window))
        cur += window
    # if the last window end < overall_end, it's already captured above

    # add one final window *anchored* on overall_end
    last_start = overall_end - window
    if not windows or last_start > windows[-1][0]:
        windows.append((last_start, overall_end))

    # now windows is a list of full‑length 1 000 h slices,
    # with the last one ending exactly at overall_end.

    # pick the slice‐th window
    if slice_idx >= len(windows):
        # out of range: no data
        return slice_idx, 0, 0
    win_start, win_end = windows[slice_idx]

    # split each track by that time span
    df1 = df_full[df_full.idcollar == id1]
    df2 = df_full[df_full.idcollar == id2]
    s1 = df1[(df1.time >= win_start) & (df1.time < win_end)]
    s2 = df2[(df2.time >= win_start) & (df2.time < win_end)]
    # only proceed if *both* have any data
    if s1.empty or s2.empty:
        return slice_idx, 0, 0

    df_slice = pd.concat([s1, s2], ignore_index=True)
    df_slice['time'] = pd.to_datetime(df_slice['time'], errors='coerce').dt.floor('S')
    print(f"[Slice {slice_idx:03d}] window {win_start}→{win_end}, n1={len(s1)}, n2={len(s2)}")
    # ───────────────────────────────────────────────────────────────────────────

    # 1b) OBSERVED‐SLICE INTERACTIONS
    valid_attrs = sorted({
        layer
        for iid in id_pair
        for layer in ssf_weights_dict[iid]
        if layer in df_slice.columns
    }) or None

    for lo, hi, lbl in zip(time_bins[:-1], time_bins[1:], time_labels):
        ev_path = int_dir / f"{id1}_{id2}_{slice_label}_events_{lbl}_observed_time.parquet"
        pr_path = int_dir / f"{id1}_{id2}_{slice_label}_pairs_{lbl}_observed_time.parquet"
        if ev_path.exists() and pr_path.exists():
            continue

        or_obs = ortega.ORTEGA(
            data             = df_slice,
            latitude_field   = "proj_lat",
            longitude_field  = "proj_lon",
            minute_min_delay = float(lo),
            minute_max_delay = float(hi),
            time_field       = "time",
            id_field         = "idcollar",
            max_el_time_min  = 121.0,
            speed_average    = True,
            attr_fields      = valid_attrs
        )
        res_obs = or_obs.interaction_analysis()
        if res_obs is None or res_obs.df_interaction_events.empty:
            continue
        if lo == 0:
            res_obs.compute_interaction_duration()

        ev = res_obs.df_interaction_events.copy()
        ev["slice"]    = slice_idx
        ev["time_bin"] = lbl
        ev.to_parquet(ev_path, index=False)

        pr = res_obs.df_all_intersection_pairs.copy()
        pr["slice"]    = slice_idx
        pr["time_bin"] = lbl
        pr.to_parquet(pr_path, index=False)

    # 2) re‐fit distributions
    slf   = analyze_step_length_distribution(df_slice,   id_pair)
    taf   = analyze_turning_angle_distribution(df_slice, id_pair)
    sf    = analyze_slope_distribution(df_slice,         id_pair)

    # 3) clip & combine rasters
    combined_probs, prob_means, combined_masks = clip_and_normalize_slope(
        df_slice, raster_paths, clip_dir, id_pair, slope_fits=sf)

    # 4) run M_C CRW sims
    for id_sel in id_pair:
        persis = taf[id_sel]["persist_dir"]
        mu      = taf[id_sel]["mean_turn_angle"]
        sigma, sigma_1 = 10, 45

        # load clipped raster
        dem_file = clip_dir / f"combined_probability_MCP_{id_sel}.tif"
        myArray, geotransform = raster_to_numpy_array(str(dem_file))
        envArr = combined_probs[id_sel]
        mask   = combined_masks[id_sel]
        if envArr.ndim==3 and envArr.shape[0]==1:
            envArr = envArr[0]

        df_obs = df_slice[df_slice.idcollar == id_sel].copy()
        df_obs["time"] = pd.to_datetime(df_obs["time"], errors="coerce")

        time_step_hours = estimate_time_step_hours(df_obs)
        time_step_hours = max(0.25, min(time_step_hours, 4.0))  # between 15 mins and 4 hours
        print(f"time_step_hours for {id_sel}: {time_step_hours:.2f}")

        t0 = pd.to_datetime(df_obs.loc[df_obs.index[0], "time"])

        _, df_mc_all = create_crw_monte_carlo_simulation(
            M_C, len(df_obs), persis, mu, sigma, sigma_1,
            myArray, myArray.shape[1], myArray.shape[0],
            envArr, mask, np.zeros_like(myArray,int),
            time_step_hours, t0, id_sel, slf, taf
        )

        # Extract environmental values and projected coordinates
        proj_lons, proj_lats = [], []  # Store projected coordinates
        for layer_name in ["slope", "bt", "gr", "sb"]:
            # Check if the individual uses this layer
            if layer_name not in ssf_weights_dict[id_sel]:
                print(f"Skipping {layer_name} for {id_sel}, not used in SSF.")
                continue
        
            # Define clipped raster path
            clipped_raster_path = f"{clip_dir}/{layer_name}_clipped_MCP_{id_sel}.tif"
        
            # Extract values only if the clipped raster exists
            env_values, proj_lons, proj_lats = extract_from_clipped_raster(df_mc_all, clipped_raster_path, geotransform)
            df_mc_all[layer_name] = env_values  # Assign extracted raster values
        
        # Assign projected coordinates to DataFrame
        df_mc_all["proj_lon"] = proj_lons
        df_mc_all["proj_lat"] = proj_lats

        outp = sim_dir / f"crw_sim_{id1}_{id2}_{slice_label}_{id_sel}.parquet"
        df_mc_all.to_parquet(outp, index=False)

    # 5) null‐model loop
    total_events = total_pairs = 0
    for MC in range(1, M_C+1):
        for lo, hi, lbl in zip(time_bins[:-1], time_bins[1:], time_labels):
            _, _, _, ne, np_ = process_slice_round(
                slice_label, MC, lo, hi, lbl,
                sim_dir, id_pair, int_dir, ssf_weights_dict
            )
            total_events += ne
            total_pairs  += np_

    return slice_idx, total_events, total_pairs



def process_slice_reverse_time(
    slice_idx: int,
    window_start: pd.Timestamp,
    window_end:   pd.Timestamp,
    df_full: pd.DataFrame,
    id_pair: Tuple[int,int],
    raster_paths: dict,
    slice_output_base: Path,
    M_C: int = 10,
    time_bins: List[float] = None,
    time_labels: List[str] = None,
    mc_start: int = 1,
    mc_stop: Optional[int] = None
) -> Tuple[int,int,int]:
    """
    1) Cut out slice #slice_idx (1 000 hours window)
    2) Re‐fit all distributions on that slice
    3) Clip & combine rasters
    4) Run M_C CRW sims → write two per‐ID Parquets
    5) For each (MC_round, time‐bin), call process_slice_round()
    Returns (slice_idx, total_events, total_pairs)
    """

    id1, id2 = id_pair
    slice_label = f"slice{slice_idx:03d}"
    if mc_stop is None:
        mc_stop = M_C

    # prepare output folders
    clip_dir = slice_output_base / slice_label / "clipped_rasters"
    sim_dir  = slice_output_base / slice_label / "simulations"
    int_dir  = slice_output_base / slice_label / "interactions"
    for d in (clip_dir, sim_dir, int_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────────────
    # 1) REVERSE 1 000‑HOUR WINDOWS ANCHORED ON THE END, THEN REVERSE TO CHRONOLOGICAL
    overall_start = df_full['time'].min()
    overall_end   = df_full['time'].max()
    window        = pd.Timedelta(hours=1000)

    # build windows anchored at the end
    rev_windows = []
    cur_end = overall_end
    while True:
        cur_start = cur_end - window
        if cur_start < overall_start:
            # if the last head‐chunk is shorter than 1000 h, drop it
            break
        rev_windows.append((cur_start, cur_end))
        cur_end = cur_start

    # rev_windows = [(last-1000h, last), (last-2000h, last-1000h), …]
    # reverse to chronological so slice000 is the earliest window
    windows = list(reversed(rev_windows))

    # pick the slice‐th window
    if slice_idx >= len(windows):
        return slice_idx, 0, 0
    win_start, win_end = windows[slice_idx]

    # split by that time span
    df1 = df_full[df_full.idcollar == id1]
    df2 = df_full[df_full.idcollar == id2]
    s1 = df1[(df1.time >= win_start) & (df1.time < win_end)]
    s2 = df2[(df2.time >= win_start) & (df2.time < win_end)]
    if s1.empty or s2.empty:
        return slice_idx, 0, 0

    df_slice = pd.concat([s1, s2], ignore_index=True)
    df_slice['time'] = pd.to_datetime(df_slice['time'], errors='coerce').dt.floor('S')
    print(f"[Slice {slice_idx:03d}] window {win_start} → {win_end}, n1={len(s1)}, n2={len(s2)}")
    # ───────────────────────────────────────────────────────────────────────────

    # 1b) OBSERVED‐SLICE INTERACTIONS
    for lo, hi, lbl in zip(time_bins[:-1], time_bins[1:], time_labels):
        ev_path = int_dir / f"{id1}_{id2}_{slice_label}_events_{lbl}_observed_time.parquet"
        pr_path = int_dir / f"{id1}_{id2}_{slice_label}_pairs_{lbl}_observed_time.parquet"
        if ev_path.exists() and pr_path.exists():
            continue

        or_obs = ortega.ORTEGA(
            data             = df_slice,
            latitude_field   = "proj_lat",
            longitude_field  = "proj_lon",
            minute_min_delay = float(lo),
            minute_max_delay = float(hi),
            time_field       = "time",
            id_field         = "idcollar",
            max_el_time_min  = 121.0,
            speed_average    = True,
            attr_fields      = None
        )
        res_obs = or_obs.interaction_analysis()
        if res_obs is None or res_obs.df_interaction_events.empty:
            continue
        if lo == 0:
            res_obs.compute_interaction_duration()

        ev = res_obs.df_interaction_events.copy()
        ev["slice"]    = slice_idx
        ev["time_bin"] = lbl
        ev.to_parquet(ev_path, index=False)

        pr = res_obs.df_all_intersection_pairs.copy()
        pr["slice"]    = slice_idx
        pr["time_bin"] = lbl
        pr.to_parquet(pr_path, index=False)

    # 2) re‐fit distributions
    slf   = analyze_step_length_distribution(df_slice,   id_pair)
    taf   = analyze_turning_angle_distribution(df_slice, id_pair)
    sf    = analyze_slope_distribution(df_slice,         id_pair)

    # 3) clip & combine rasters
    combined_probs, prob_means, combined_masks = clip_and_normalize_slope(
        df_slice, raster_paths, clip_dir, id_pair, slope_fits=sf)

    # 4) run M_C CRW sims
    for id_sel in id_pair:
        persis = taf[id_sel]["persist_dir"]
        mu      = taf[id_sel]["mean_turn_angle"]
        sigma, sigma_1 = 10, 45

        # load clipped raster
        dem_file = clip_dir / f"combined_probability_MCP_{id_sel}.tif"
        myArray, geotransform = raster_to_numpy_array(str(dem_file))
        envArr = combined_probs[id_sel]
        mask   = combined_masks[id_sel]
        if envArr.ndim==3 and envArr.shape[0]==1:
            envArr = envArr[0]

        df_obs = df_slice[df_slice.idcollar == id_sel].copy()
        df_obs["time"] = pd.to_datetime(df_obs["time"], errors="coerce")

        time_step_hours = estimate_time_step_hours(df_obs)
        time_step_hours = max(0.25, min(time_step_hours, 4.0))  # between 15 mins and 4 hours
        print(f"time_step_hours for {id_sel}: {time_step_hours:.2f}")

        t0 = pd.to_datetime(df_obs.loc[df_obs.index[0], "time"])

        _, df_mc_all = create_crw_monte_carlo_simulation(
            M_C, len(df_obs), persis, mu, sigma, sigma_1,
            myArray, myArray.shape[1], myArray.shape[0],
            envArr, mask, np.zeros_like(myArray,int),
            time_step_hours, t0, id_sel, slf, taf, mc_start = mc_start
        )

        # Extract environmental values and projected coordinates
        slope_path = clip_dir / f"slope_clipped_MCP_{id_sel}.tif"
        vals, proj_lons, proj_lats = extract_from_clipped_raster(
            df_mc_all, str(slope_path), geotransform
        )
        df_mc_all["slope"]    = vals
        df_mc_all["proj_lon"] = proj_lons
        df_mc_all["proj_lat"] = proj_lats

        outp = sim_dir / f"crw_sim_{id1}_{id2}_{slice_label}_{id_sel}.parquet"
        df_mc_all.to_parquet(outp, index=False)

    # 5) null‐model loop
    total_events = total_pairs = 0
    for MC in range(mc_start, mc_stop+1):
        for lo, hi, lbl in zip(time_bins[:-1], time_bins[1:], time_labels):
            _, _, _, ne, np_ = process_slice_round(
                slice_label, MC, lo, hi, lbl,
                sim_dir, id_pair, int_dir
            )
            total_events += ne
            total_pairs  += np_

    return slice_idx, total_events, total_pairs
