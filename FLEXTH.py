################################################ 
################################################  
#     / ____// /    / ____/ |/ //_  __// / / / #
#    / /_   / /    / __/  |   /  / /  / /_/ /  #
#   / __/  / /___ / /___ /   |  / /  / __  /   #
#  /_/    /_____//_____//_/|_| /_/  /_/ /_/    #
################################################                                         
################################################                                 
                                                              
# WELCOME TO FLEXTH - THE FLOOD EXTENT ENHANCEMENT AND WATER DEPTH ESTIMATION 
# TOOL FOR (SATELLITE-DERIVED) INUNDATION MAPS

# The tool enhances flood delineation maps (e.g. satellite-derived) by extending 
# floods based on terrain topography and additional optional information. 
# The algorithm employs topographical data (in the form of a DTM) 
# in combination with flood delineations to provide water depth estimates.
# The algorithm requires, as a primary input, a flood delineation map and a DTM. 
# Additional information may include areas excluded from flood mapping (i.e. no data)
# and/or permanent water bodies. 
# Input delineations must be provided via binary (i.e. 0/1) georeferenced rasters 
# (GeoTIFF) in a suitable PROJECTED reference system.
# Since water depth is the primary proxy for flood damages, the tool aims to facilitate
# flood impact assessment over large scales with minimum supervision and computational times.
# 
# The script 'DTM_2_floodmap.py' helps to easily bring your DTM (or any other input) into 
# the same extent,  resolution, grid and projections  as the input flood map raster.

# For further details see: 
#'Water depth estimate and flood extent enhancement for satellite-based inundation maps' 
# by Betterle and Salamon (2024) on Natural Hazards and Earth System Sciences, 
# https://doi.org/10.5194/nhess-24-2817-2024,
#
# visit https://code.europa.eu/floods/floods-river/flexth
#
# or contact andrea.betterle@ec.europa.eu or peter.salamon@ec.europa.eu
#
# ---------------------------------------------------------------------------
# MODIFIED VERSION (uncertainty-aware FLEXTH)
# This file is a modified version of FLEXTH v1.3.0 (EUPL v1.2, (C) European Union).
# Changes: an optional pixel-wise uncertainty gate restricts where flood water
# may propagate; the water-level interpolation itself is unchanged. Changes are
# tagged "UNCERTAINTY-FLEXTH CHANGE" in the code and described in
# docs/UNCERTAINTY_GATE.md.
# ---------------------------------------------------------------------------


##############################################################################
##############################################################################
##############################################################################

##########################
# IMPORT WORKING LIBRARIES  
##########################

import numpy as np
import rasterio
from rasterio.merge import merge
import cv2
from astropy.convolution import convolve
import warnings
import os, glob
import time
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib
from rasterio.windows import Window
from pathlib import Path
import imageio.v2 as imageio
matplotlib.use('Agg') 
######################
#INPUTs and PARAMETERS 
######################


#   INPUT
#   a folder named 'input_dir'  must contain the following maps in a suitable PROJECTED reference system :
    
#   flood.tif            :   binary map delineating flooded areas (0: no flood; 1:flood)
#   dtm.tif              :   digital terrain model
#   exclusion.tif        : * binary map delineating areas excluded from flood mapping (0: no mask; 1:mask (i.e. no-data))
#   permanent_water.tif  : * binary map delineating permanent/seasonal water bodies (0: no water; 1:water)
#   obswater.tif         : * binary map delineating all observed water (union between flood and permanent_water) - to be used if permanent_water.tif is not provided (0: no water; 1:water)
#   (*:  optional input)
#
#  OUTPUT
#  output files generataed by FLEXTH:
#
#  WD.tif :  flood extent with water depth estimates (in cm)
#  WL.tif :  flood extent with water level estimates (in the vertical coordinate of the DTM, typically m a.s.l.)
#
#
#  Below, all variable whose names start with "param" denote adjustable parameters.
#  The most relevant are listed below whereas throughout the codes there are 
#  additional less sensitive and more geeky ones. Testing showed that the 
#  default values are effective and robust in a wide range of settings. 
#  Nonetheless, parameters can be tweaked to match specific needs and/or use cases.
#  Parameters may require adjustments for resolutions larger/smaller than 20 m and/or 
#  depending on the accuracy of the flood delineation and/or of the DTM


# TILING
# If param_tiling = True, the inputs are tiled and FLEXTH is run in sequence across the tiles. 
# The tiled rasters are stored in the authomatically created subfolder "input_tiled".
# Tiling can be necessary for analysis on large rasters (or high resolutions) when computational resources are limited.
# Smaller tiles use less resources but might generate some artifacts in flooded areas across the borders of the tiles. 
#
# If you want to run FLEXH in tiling mode, set BOTH 'param_tiling=True' AND 'param_tile_inputs=True' during the first run. 
# If you  want to re-run FLEXTH (without the need to re-tile the inputs) you can set 'param_tile_inputs=False'. 

#see the 'Parameters' section in https://code.europa.eu/floods/floods-river/flexth for more details 

param_tiling      = False   # True to run FLEXTH on the tiled inputs
param_tile_inputs = True    # True to tile the inputs. If input is already tiled, select False. Relevant if "param_tiling = True"
param_tile_size   = 10000   # size of the squared tiles (pixels)
param_merge_tiles = True    # True to automatically merge the tiled outputs in the end


# Select the output that will be generated: Water depth ("WD") , Water level ("WL"), both ("WL_WD")
param_output_map = 'WL_WD'


input_dir  = Path(r'./input')
output_dir = Path(r'./output')

flood_path = "./input/flood.tif"
dtm_path = "./input/dtm.tif"
exclusion_path = "./input/exclusion.tif"
permanent_water_path = "./input/permanent_water.tif"
obswater_path = "./input/obswater.tif"
uncertainty_path = "./input/uncertainty.tif"  # required when uncertainty_on = 1
classification_path = "./input/classification.tif"  # required for semantic_high_uncertainty
water_probability_path = "./input/water_probability.tif"  # required for semantic_high_uncertainty
retention_curve = {90:0.31842,80:0.14068,70:0.088596,60:0.06533,50:0.0506287}
uncertainty_threshold = float(retention_curve[70])  # threshold interpreted according to uncertainty_gate_mode
uncertainty_on = 1
# UNCERTAINTY-FLEXTH CHANGE U1: use EDL class/probability semantics; set
# "legacy_low_uncertainty" to reproduce the earlier unc < threshold gate.
uncertainty_gate_mode = "semantic_high_uncertainty"


# WATER LEVEL ESTIMATION METHOD (options: 'method_A', 'method_B'): 
param_WL_estimation_method = 'method_A'  



###############
# PARAMETERS ##
###############


# --code name ----------------------- value---- name -- range ---- default- units -- description---------------------------------------
param_threshold_slope               =  0.1     # S_max [0, 1]        {0.1}  (-)  : border pixels steeper than this (D_z/D_x) are not used to estimate water level
param_size_gaps_close               =  0.1     # A_g   [0, 1]        {0.1}  (km2): gaps up to this size in the initial flood map will be closed

param_max_number_neighbors          =  200     # N_max [1, 1000]     {200}  (-):   number of border pixels used to compute water level at each location
param_inverse_dist_exp              =  1       # alpha [0, 5]        {1}    (-):   inverse distance weighting exponent used to interpolate WL inside flooded areas 
param_border_subsampling            =  0       # Sbs_r [0, 100]:     {0}    (-):   parameter tuning the subsampling rate of border pixels in large flooded areas

param_min_flood_border_size         =  10      # N_min [1, 1000]     {10}   (-):   if the number of valid pixels along the border is less than this, WL is estimated based on the distribution of the elevation of the pixels inside the flooded area
param_border_quantile               =  0.5    # P     [0, 1]        {0.5}  (-):   assign water level based on the quantile P of the elevation of border cells (valid if Method B is selected)
param_inner_quantile                =  0.98    # P*    [0, 1]        {0.98} (-):   in flooded areas with less that N_min valid border pixels, uses this quantile of the elevation of the pixels inside the flooded area to estimate water level

param_spread_outside_exclusion_mask = False    #Sprd   [True, False] {False}(-):   if True, flood propagation is performed also outside the exclusion mask (i.e. outside no-data areas) 
param_max_propagation_distance      = 10      # D_max [0, 100]      {10}   (km):  maximum propagation distance for an arbitrarily large flooded area
param_distance_range                = 10       # A_1/2 [0, 100]      {10}   (km2): flooded areas of this size can reach half of the maximum propagation distance  
    
param_WD_star                       = 10      # WD*   [0, 100]:     {10}   (cm) : dummy water depth assigned where estimated WL is below ground surface (i.e WL < DTM) 




#######################################################################################
#######################################################################################
## END INPUT - NO NEED TO MODIFY ANYTHING AFTER THIS POINT (UNLESS YOU HAVE GOOD IDEAS)
#######################################################################################
#######################################################################################



##############
#FUNCTIONS######
##################

#TILES INPUT_RASTER INTO OUTPUT_DIR
def tiling(input_raster, source_dir, param_tile_size):
    """
    Tiles an input raster into smaller segments and saves them to a specified directory.

    This function takes a large GeoTIFF file and divides it into smaller, manageable tiles 
    based on a specified tile size. Each tile is saved as an individual GeoTIFF file in 
    an 'input_tiled' subdirectory within the provided source directory.

    Parameters:
    input_raster (str or Path): The path to the input GeoTIFF file that needs to be tiled.
    source_dir (str or Path): The directory where the 'input_tiled' folder will be created 
                              to store the output tile files.
    param_tile_size (int): The size of each tile in pixels, defining the width and height 
                           of the square tiles.

    Returns:
    None: The function saves the output tiles directly to disk and does not return any values.
    """

    from rasterio.windows import Window
    
    file_name     = os.path.splitext(os.path.basename(input_raster))[0]
    output_folder = source_dir / 'input_tiled'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the GeoTIFF file
    with rasterio.open(input_raster) as src:
        # Get the raster size
        rastersize_width, rastersize_height = src.width, src.height
    
        # Calculate the number of tiles in x and y directions 向上取整 整數除法計算完整覆蓋的數量
        num_tiles_x = (rastersize_width  + param_tile_size - 1) // param_tile_size
        num_tiles_y = (rastersize_height + param_tile_size - 1) // param_tile_size
    
        # Loop through each tile and extract it from the original raster
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Calculate the tile's bounding box
                x_off = i * param_tile_size
                y_off = j * param_tile_size
                x_size = min(param_tile_size, rastersize_width  - x_off)
                y_size = min(param_tile_size, rastersize_height - y_off)
    
                # Create a window object for the tile
                window = Window(x_off, y_off, x_size, y_size)
    
                # Read the tile from the original raster
                tile_data = src.read(1, window=window)
    
                # Create a new GeoTIFF file for the tile
                tile_profile = src.profile
                tile_profile.update({
                    'width': x_size,
                    'height': y_size,
                    'transform': rasterio.windows.transform(window, src.transform),
                    'compress': 'deflate'
                })
                with rasterio.open(output_folder/f'{file_name}_tile_{i+1}_{j+1}.tif', 'w', **tile_profile) as dst:
                    dst.write(tile_data, 1)
    
    
    
#IDENTIFY THE INDICES OF NEIGHBORING CELLS IN A 2D ARRAY BASED ON SPECIFIED CONNECTIVITY
def ij_neighbors(i,j,n_row,n_col,connectivity):
    """
    This function calculates the indices of neighboring cells for a given cell in a 2D array.
    The type of connectivity (4, 8, or 24) determines which neighbors are considered. The 
    function ensures that neighbors are within the bounds of the array by adjusting indices 
    for edge cases.

    Parameters:
    i (int): The row index of the target cell.
    j (int): The column index of the target cell.
    n_row (int): The total number of rows in the array.
    n_col (int): The total number of columns in the array.
    connectivity (int): The type of connectivity to use, which can be:
        - 4: Only directly adjacent horizontal and vertical neighbors.  只有上下左右  4
        - 8: All adjacent neighbors, including diagonals.  上下左右和四個斜角  3*3-1=8
        - 24: A larger neighborhood, including two cells away in all directions. 兩格範圍內的所有鄰居   5*5*-1=24

    Returns:
    np.ndarray: A 2D NumPy array of shape (N, 2), where N is the number of neighbors.
                Each row represents the (row, column) indices of a neighboring cell.

    Raises:
    ValueError: If an unsupported connectivity value is provided.
    """


    if connectivity == 4:
        neighbors=np.array([ [i-1,j], [i,j+1], [i+1,j],  [i,j-1]])    
    elif connectivity == 8:
        neighbors=np.array([[i-1,j-1],[i-1,j], [i-1,j+1], [i,j+1], [i+1,j+1], [i+1,j], [i+1,j-1], [i,j-1]])  
    elif connectivity == 24:
        neighbors=np.array([[i-2, j-2],[i-2, j-1], [i-2,j], [i-2,j+1], [i-2,j+2], [i-1, j-2], [i-1,j-1], [i-1,j], [i-1,j+1], [i-1,j+2], [i,j-2], [i,j-1], [i,j+1], [i,j+2], [i+1,j-2], [i+1,j-1], [i+1,j], [i+1,j+1],[i+1,j+2],[i+2,j-2],[i+2,j-1], [i+2,j], [i+2,j+1], [i+2,j+2]  ]     ) 
    else:
        raise ValueError(f"Unsupported connectivity value: {connectivity}")
    
    for indice in range(len(neighbors)):
        if neighbors[indice,0]> n_row-1 or neighbors[indice,0]<0 :  #檢查最後一行索引是否大於總行數-1 或者最上一行索引是否小於0
            neighbors[indice,0]=i                       #將超出邊界的索引重設為原始索引
            neighbors[indice,1]=j
        if neighbors[indice,1]> n_col-1 or neighbors[indice,1]<0 :  #檢查最後一列索引是否大於總列數-1 或者最左列索引是否小於0
            neighbors[indice,0]=i
            neighbors[indice,1]=j                     
            
    return neighbors.astype('uint32')    
   
    

#COMPUTES THE WEIGHTED QUANTILES OUT OF A PDF
def weighted_quantile(values, percentile, weights=None):
    """
    This function calculates weighted quantiles for a given dataset, similar to 
    `numpy.percentile`, but with support for weights. It is particularly useful 
    for computing quantiles of data where different elements have different 
    levels of importance or frequency.

    Parameters:
    values (np.ndarray): A 2D NumPy array with data for which to compute quantiles.
    percentile (float or np.ndarray): The quantile(s) to compute, expressed as a 
                                      fraction between 0 and 1 (inclusive).
    weights (array-like, optional): An array of weights associated with the values.
                                    Must be the same shape as `values`. If None, 
                                    equal weight is assumed for all elements.

    Returns:
    np.ndarray: A 1D NumPy array containing the computed weighted quantile for 
                each row of the input data.

    Raises:
    AssertionError: If any percentile values are not within the range [0, 1].
    """

    values = np.array(values)
    percentile = np.array(percentile)
    if weights is None:
        weights= np.ones(np.shape(values))
    weights = np.array(weights)
    assert np.all(percentile >= 0) and np.all(percentile <= 1), \
        'percentile should be in [0, 1]'

    sorter = np.argsort(values)
    values = np.take_along_axis(values, sorter, axis = 1)   #重新排序成遞增
    weights = np.take_along_axis(weights, sorter, axis = 1)      #同樣的排序對應到 weights 確保 weights 與 values 對應

    weighted_quantiles  = np.cumsum(weights, axis = 1 ) - 0.5 * weights #沿著指定軸將所有 weights 的累積和，再扣掉 0.5 * weights使權重的中點對應到 values
    weighted_quantiles /= np.sum(weights, axis = 1)[:, np.newaxis] * np.ones ( np.shape(weighted_quantiles)) #將每一行的累積和除以該行的總權重，將權重標準化到 [0, 1] 範圍內，當中有擴展至相同形狀的矩陣
    
    return values [np.array( np.arange(0,np.shape(values)[0],1)  ), np.argmin( (weighted_quantiles - percentile  )**2, axis = 1) ] 
#argmin 沿著指定的軸，找到最小值所在的索引； np.arange 產生一個從 0 到 np.shape(values)[0]-1 的整數陣列，代表每一行的索引
#weighted_quantiles - percentile 計算每個加權分位數與目標分位數之間的差異，並將其平方以確保所有值為正數
#回傳該點在原始數值陣列values中對應的值(尋找最小是因為最接近目標的分位數)

##################
##################
# MAIN FUNCTION ##
##################
##################

def build_semantic_candidate_mask(
    classification,
    water_probability,
    uncertainty,
    threshold,
):
    """Return pixels where optical semantics allow terrain-based propagation."""
    invalid = (
        (classification == 0)
        | (~np.isfinite(water_probability))
        | (water_probability < 0)
    )
    # Cloud (class 3) is unobserved like invalid data: the water head was not
    # trained under bright clouds, so its uncertainty there is not used.
    cloud = classification == 3
    high_uncertainty = (
        np.isfinite(uncertainty)
        & (uncertainty >= 0)
        & (uncertainty >= threshold)
    )
    uncertain_land = (classification == 1) & high_uncertainty
    # Class 4 was first predicted as water (p_water > 0.5), then relabeled
    # flood_trace by MNDWI. It is terrain-eligible but is not an initial seed.
    flood_trace_candidate = classification == 4
    return (
        invalid
        | cloud
        | uncertain_land
        | flood_trace_candidate
    )


def flood_processing(
    flood_path,
    dtm_path,
    uncertainty_path,
    classification_path,
    water_probability_path,
    exclusion_path,
    obswater_path,
    permanent_water_path,
):
           
    # IMPORT FLOOD MAP AND DTM
    with rasterio.open(flood_path) as src:
        flood     = src.read(1).astype('uint8')
        flood[(flood !=1) & (flood !=0)]  = 0   #將非0非1的值設為0
        transform = src.transform  #將影像的像素座標轉換成實際地理座標
        crs       = src.crs    #座標參考系

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)
        transform_dtm = src.transform
        crs_dtm       = src.crs
        dtm_nodata    = src.nodata
        dtm[dtm == dtm_nodata]  = np.nan  #將無效值設為nan
    
    
    #RISE AN ERROR IF THE SPATIAL REFERENCES ARE NOT MATCHING
    if transform != transform_dtm or crs != crs_dtm:  #如果像素轉換方式或座標參考系不同，出現error
        raise TypeError("Flood map and DTM don't share the same projections and/or grid!")
    
    if uncertainty_on and not os.path.isfile(uncertainty_path):
        raise FileNotFoundError(
            f"uncertainty_on=1 but uncertainty input is missing: {uncertainty_path}"
        )

    if os.path.isfile(uncertainty_path):  #檢查是否有提供不確定性的遮罩檔案
        with rasterio.open(uncertainty_path) as src:
            uncertainty = src.read(1).astype(np.float32)
            transform_uncertainty = src.transform
            crs_uncertainty       = src.crs
            
            if (
                transform != transform_uncertainty
                or crs != crs_uncertainty
                or uncertainty.shape != flood.shape
            ):
                raise TypeError("Flood map and uncertainty mask don't share the same projections and/or grid!")

    valid_gate_modes = {
        "legacy_low_uncertainty",
        "semantic_high_uncertainty",
    }
    if uncertainty_gate_mode not in valid_gate_modes:
        raise ValueError(
            f"Unknown uncertainty_gate_mode={uncertainty_gate_mode!r}; "
            f"expected one of {sorted(valid_gate_modes)}"
        )

    semantic_candidate_mask = None
    if uncertainty_on and uncertainty_gate_mode == "semantic_high_uncertainty":
        # UNCERTAINTY-FLEXTH CHANGE U2: load EDL class and water probability.
        missing_semantic_inputs = [
            str(path)
            for path in (classification_path, water_probability_path)
            if not os.path.isfile(path)
        ]
        if missing_semantic_inputs:
            raise FileNotFoundError(
                "Semantic uncertainty gate requires: "
                + ", ".join(missing_semantic_inputs)
            )

        with rasterio.open(classification_path) as src:
            classification = src.read(1).astype(np.uint8)
            if (
                transform != src.transform
                or crs != src.crs
                or classification.shape != flood.shape
            ):
                raise TypeError(
                    "Flood map and EDL classification don't share the same "
                    "projection and/or grid!"
                )

        with rasterio.open(water_probability_path) as src:
            water_probability = src.read(1).astype(np.float32)
            if (
                transform != src.transform
                or crs != src.crs
                or water_probability.shape != flood.shape
            ):
                raise TypeError(
                    "Flood map and water probability don't share the same "
                    "projection and/or grid!"
                )

        semantic_candidate_mask = build_semantic_candidate_mask(
            classification,
            water_probability,
            uncertainty,
            uncertainty_threshold,
        )

    #CHECKS IF OPTIONAL INPUTS ARE PROVIDED AND IMPORT THEM
    if os.path.isfile(exclusion_path): #檢查是否有提供沒有資料的遮罩檔案
        with rasterio.open(exclusion_path) as src:
            exclusion           = src.read(1).astype('uint8')
            exclusion[(exclusion !=1) & (exclusion !=0)] = 1 #將非0非1的值設為1
            transform_exclusion = src.transform
            crs_exclusion       = src.crs
    
            if transform != transform_exclusion or crs  != crs_exclusion:
                raise TypeError("Flood map and exclusion mask don't share the same projections and/or grid!")
    else:
        exclusion  = np.full_like(flood, 0)   #如果沒有提供遮罩檔案，建立一個與flood相同大小的全0陣列
        

    if os.path.isfile(obswater_path): #檢查是否有提供觀測水體的遮罩檔案
        obswater_from_flood = False
        with rasterio.open(obswater_path) as src:
            obswater           = src.read(1).astype('uint8')
            obswater[(obswater !=1) & (obswater !=0)]  = 0  #將非0非1的值設為0
            transform_obswater = src.transform
            crs_obswater       = src.crs
            
            if transform != transform_obswater or crs  != crs_obswater:
                raise TypeError("Flood map and observed water don't share the same projections and/or grid!")
                
                
    else:
        obswater_from_flood = True
        obswater  = np.copy(flood)   #如果沒有提供觀測水體的遮罩檔案，將obswater設為flood的複製
    

    if os.path.isfile(permanent_water_path): #檢查是否有提供永久水體的遮罩檔案
        with rasterio.open(permanent_water_path) as src:
            permanent_water           = src.read(1).astype('uint8')
            permanent_water[(permanent_water !=1) & (permanent_water !=0)]   = 0
            transform_permanent_water = src.transform
            crs_permanent_water       = src.crs
            
            if transform != transform_permanent_water or crs  != crs_permanent_water:
                raise TypeError("Flood map and permanent water don't share the same projections and/or grid!")            
            
    else:
        permanent_water  = np.full_like(flood, 0)    #如果沒有提供永久水體的遮罩檔案，建立一個與flood相同大小的全0陣列

    


    #GEOTRANSFORMATION
    transform*(0,0)                # goes from COL-ROW index to X-Y:  trans*(ncol,nrow)=(x_coordinate,y_coordinate)
    n_row, n_col    = flood.shape  #extract the dimension of the raster
    L = transform.a                # pixel size in m (pixels should be squared)
    #a 代表每個像素在水平方向上的實際大小 (像素應該是正方形的) L代表像素的邊長(以公尺為單位)
     
    #LOCAL SLOPE
    slope_max =  np.gradient(dtm, L)   #包含兩個陣列的list，分別代表在每個方向上的梯度(水平與垂直方向)
    slope_max =  np.sqrt(slope_max[0]**2 + slope_max[1]**2 )   #將斜邊(總坡度)計算出來存回 slope_max，變成一個二維陣列
    
    #IDENTIFIES WHERE SLOPE IS BEYOND THE THRESHOLD 
    slope_steep   =   np.array(slope_max >  param_threshold_slope).astype('uint8')
    #使用到控制坡度的參數
    
    #############################################################################
    ##CONNECTED COMPONENTS ANAYSIS WITH CV2 - IDENTIFIES CONTIGUOUS FLOODED AREAS        #辨識連續的洪水區域
    #############################################################################  
    
    #SOME KERNELS 
    kernel_1       = np.ones((3,3),np.uint8)  # 3x3的全1矩陣
    kernel_1_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)  #十字形的3x3矩陣
    kernel_2       = np.ones((5,5),np.uint8) # 5x5的全1矩陣
    
    #RUNS MORPHOLOGICAL CLOSING TO REMOVE SMALL HOLES AND VERY IRREGULAR BORDERS FROM FLOOD AND OBSWATER
    flood_seed = np.copy(flood)
    flood    = cv2.morphologyEx(flood   ,    cv2.MORPH_CLOSE, kernel_1_cross,  iterations = 2 ) #形態學閉運算
    obswater = cv2.morphologyEx(obswater,    cv2.MORPH_CLOSE, kernel_1_cross,  iterations = 2 ) #使用十字形核來(先侵蝕再膨脹)2次
    #目的是要填補孔洞與連接斷裂處，利用opencv來對flood和obswater進行形態學閉運算

    #DILATE EXCLUSION MASK
    exclusion = cv2.dilate(exclusion  ,  kernel_1) #使用3x3全1矩陣來膨脹須排除的區域(無資料、其他不應該被視為水體的區域)
    

    #############################################
    #CLOSES SMALL HOLES IN FLOOD MAP WITH  CV2 ##
    #############################################
    print ('Closing small gaps in flood map...')
    
    threshold_n_pixels                           = int(param_size_gaps_close*1e6 / L**2) #將參數轉單位(km2轉成像素數量)
    binary_map_complementary_flood               = np.logical_not(flood).astype('uint8') # flood的反轉(0變1,1變0)
    z_num_labels, z_labels, z_stats, z_centroids = cv2.connectedComponentsWithStats(binary_map_complementary_flood,connectivity=8) 
    #對反轉後的洪水(非洪水區做連通域分析)，設定connectivity=8(包含對角線)
    # z_num_labels: 連通域的數量 (會將背景視為一個連通域)，所以實際找到的是 z_num_labels - 1 個連通域
    # z_labels: 每個像素所屬的連通分量標籤 (0表示背景，1表示第一個連通分量，依此類推) 與輸入影像相同大小的二維陣列
    # z_stats: 每個連通域的統計資訊 (包含邊界框、面積等) 是一個二維陣列，每一列代表一個連通域的統計資訊
    # [x_top, y_top, width, height, area] - 第一列是背景值(即0)
    # x_top和y_top是邊界框的左上角座標，width和height是邊界框的寬度和高度，area是連通域的像素數量
    # z_centroids: 每個連通域的質心座標 (x, y) 是一個二維陣列，每一列代表一個連通域的質心座標
    
    for i in range(len(z_stats)):
        if np.abs(z_stats[i,4]) < threshold_n_pixels:  #小於設定閾值
            flood    [ z_stats[i,1]: z_stats[i,1] + z_stats[i,3] , z_stats[i,0]:z_stats[i,0] + z_stats[i,2] ] [z_labels[ z_stats[i,1]: z_stats[i,1] + z_stats[i,3] , z_stats[i,0]:z_stats[i,0] + z_stats[i,2] ] ==i] = 1
            obswater [ z_stats[i,1]: z_stats[i,1] + z_stats[i,3] , z_stats[i,0]:z_stats[i,0] + z_stats[i,2] ] [z_labels[ z_stats[i,1]: z_stats[i,1] + z_stats[i,3] , z_stats[i,0]:z_stats[i,0] + z_stats[i,2] ] ==i] = 1
    # z_stats[i,1]: z_stats[i,1] + z_stats[i,3] 代表 y 的範圍 (從 y_top 到 y_top + height)
    # z_stats[i,0]: z_stats[i,0] + z_stats[i,2] 代表 x 的範圍 (從 x_top 到 x_top + width)
    # flood[ ... ][z_labels[ ... ] == i] = 1 將 flood 中對應於第 i 個連通域的像素設為 1 (表示洪水)
    # 後面的[]是布林遮罩，篩選出屬於第 i 個連通域的像素位置，並將這些位置的 flood 值設為 1

    if semantic_candidate_mask is not None:
        # UNCERTAINTY-FLEXTH CHANGE U3: morphology/gap filling may add seeds,
        # but those additions must not bypass the semantic propagation gate.
        flood[flood_seed == 1] = 1
        preprocessing_additions = (flood == 1) & (flood_seed == 0)
        flood[preprocessing_additions & (~semantic_candidate_mask)] = 0
        if obswater_from_flood:
            obswater = np.copy(flood)

    if os.path.isfile(input_dir / 'permanent_water.tif'):
        permanent_water = cv2.morphologyEx(permanent_water,    cv2.MORPH_CLOSE, kernel_1_cross,  iterations = 2 )
        
    else:
        permanent_water = obswater - flood  #如果沒有提供永久水體的遮罩檔案，將永久水體設為觀測水體減去洪水(即剩下的部分)
        permanent_water[(permanent_water!=0) & (permanent_water!=1) ] = 0 #將非0非1的值設為0
    


    # Applying cv2.connectedComponents() - possibly change the connectivity type 
    # !! opencv with connectivity = 4 may have problems in some versions of the package better use connectivity = 8 !!!!
    # z_stats contains: [xtop, ytop, xwidth, ywidth, #pixels] - first element of stats correspondes to background value (i.e. 0)
    
    z_num_labels, z_labels, z_stats, z_centroids = cv2.connectedComponentsWithStats(flood,connectivity=8)   
    
    flood_dilated           =   cv2.dilate(flood           ,  kernel_1)
    flood_eroded            =   cv2.erode (flood           ,  kernel_1_cross)
    # 膨脹用3x3全1矩陣，侵蝕用十字形3x3矩陣
    # 因為方形膨脹來確保小縫隙被填滿，十字侵蝕避免過度削掉角落以減少改變形狀

    exclusion_dilated       =   cv2.dilate(exclusion       ,  kernel_1)

    permanent_water_dilated =   cv2.dilate(permanent_water ,  kernel_1)
    
    
    flood_border           =   flood_dilated    - flood_eroded 
    flood_border_inner     =   flood            - flood_eroded
     
    slope_steep_dilated      =   cv2.dilate(slope_steep  ,  kernel_1)
    
    exclusion_dilated      [ (exclusion       ==0)  & (exclusion_dilated       ==1) & (flood==0) & (flood_dilated==1)] = 0
    permanent_water_dilated[ (permanent_water ==0)  & (permanent_water_dilated ==1) & (flood==0) & (flood_dilated==1)] = 0
    slope_steep_dilated    [ (slope_steep     ==0)  & (slope_steep_dilated     ==1) & (flood==0) & (flood_dilated==1)] = 0
    # 原本不是洪水區域，但膨脹後變成洪水區域，且原本的洪水區域(flood)是0，膨脹後的洪水區域(flood_dilated)是1，則將這些位置的排除遮罩、永久水體遮罩和陡坡遮罩設為0
    # 確保排除區、永久水體、陡坡不會互相干擾，以避免將未淹水點誤判為邊界點

    flood_border          =     flood_border *  np.logical_not( exclusion_dilated )   *  np.logical_not( permanent_water_dilated)   *    np.logical_not(slope_steep_dilated)  
    # 將 flood_border 中對應於排除區、永久水體和陡坡的位置設為0，保留其他位置的值

    dtm_border_flood      =   dtm * flood_border
      
   
    ## AVERAGES THE DEM ALONG THE BORDER 
    dtm_border_flood_smooth= convolve ( dtm_border_flood, kernel_2, mask = flood_border == 0, preserve_nan = True )
    dtm_border_flood_smooth[flood_border==0] = np.nan     
    # 使用5x5全1矩陣對 dtm_border_flood 進行卷積運算，並在 flood_border 為0的位置遮罩掉 (mask = flood_border == 0)
    # preserve_nan = True 參數確保在卷積過程中保持 NaN 值不變
    # 最後將 flood_border 為0的位置在 dtm_border_flood_smooth 中設為 NaN，確保這些位置不影響後續的水位估計
    
    flood_border =   flood_border * flood_border_inner


    #######################################################
    #######################################################
    ## WATER LEVEL ESTIMATION IN INITIALLY FLOODED AREAS ##
    #######################################################
    #######################################################

    z_water_level = np.full([n_row,n_col], np.nan, dtype='float32')
    # 初始化一個與輸入影像大小相同的二維陣列，並將所有元素設為 NaN，資料型態為 float32
    ROW, COL   =  np.mgrid[0:n_row,0:n_col].astype('uint32')
    # 產生兩個二維陣列 ROW 和 COL，分別代表每個像素的行索引和列索引，資料型態為 uint32

    row_flood                          =   np.ma.masked_array(ROW,      z_labels==0).compressed()
    col_flood                          =   np.ma.masked_array(COL,      z_labels==0).compressed()
    flood_labels_compressed            =   np.ma.masked_array(z_labels, z_labels==0).compressed()
    # (z_labels==0 是非洪水區)是背景值(洪水區)的遮罩條件
    # 當中未被遮蓋的元素會被保留，並且壓縮成一維陣列

    row_flood_border                   =   np.ma.masked_array(ROW,                     flood_border==0).compressed()
    col_flood_border                   =   np.ma.masked_array(COL,                     flood_border==0).compressed()
    flood_borders_labels_compressed    =   np.ma.masked_array(z_labels,                flood_border==0).compressed()
    dtm_border_flood_smooth_compressed =   np.ma.masked_array(dtm_border_flood_smooth, flood_border==0).compressed()
    # (flood_border==0 是非洪水邊界區)是非洪水邊界區的遮罩條件
    # 當中未被遮蓋的元素會被保留，並且壓縮成一維陣列

    param_border_subsampling_sill       =  100000       # Sbs_s [10, +inf]: maximum number of pixels along the border of a single flooded area used by FLEXTH to estimate water level 
    #單一淹水區域邊界點的最大使用數量(當這個淹水區域邊界像素超過這個值，會啟用二次取樣來減少計算量)

    #  ASSIGNS WATER LEVELS TO INITIAL FLOODED AREAS BASED ON THE DTM VALUES ALONG THE BORDERS
    #  OF FLOODED AREAS THAT ARE NOT CONTIGUOUS WITH THE EXCLUSION MASK
    print('Assigning water elevation to initial flooded areas...')
    
    for i in range (1,len(z_stats)):

        temp_position_flood           =  np.stack( ( row_flood[flood_labels_compressed==i],col_flood[flood_labels_compressed==i]),axis = 1)
        temp_position_border          =  np.copy(  np.stack( ( row_flood_border[flood_borders_labels_compressed==i],col_flood_border[flood_borders_labels_compressed==i]),axis = 1))
        temp_dtm_border_flood_smooth  =  np.copy( dtm_border_flood_smooth_compressed[flood_borders_labels_compressed==i])
        # 當前處理單一淹水區域i，選取當前標籤i的淹水區域和淹水邊界區域的行列索引，並將它們堆疊成二維陣列
        # temp_position_flood: 當前淹水區域i的所有像素位置 (行, 列)
        # temp_position_border: 當前淹水區域i的邊界
        # temp_dtm_border_flood_smooth: 當前淹水區域i的邊界對應的平滑後的數值高程模型 (DTM) 值

        len_border  = len(temp_position_border)
        # 計算當前淹水區域i的邊界像素數量

        # if the number of valid border pixels is less than "param_min_flood_border_size" uses the statistics of the DTM inside the flooded area
        if  len_border < param_min_flood_border_size: 
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                z_water_level[temp_position_flood[:,0], temp_position_flood[:,1] ]  =  np.nanquantile(dtm[temp_position_flood[:,0], temp_position_flood[:,1] ], param_inner_quantile)
        # 如果邊界像素數量小於設定的最小值，則使用淹水區域內的 DTM 值來估計水位

        else:   
            
            #SUBSAMPLING OF THE BORDER
            tt                            = len_border / param_border_subsampling_sill
            param_border_subsampling_     = 2 - param_border_subsampling 
            param_border_subsampling_     = param_border_subsampling_ if param_border_subsampling_ != 0  else (param_border_subsampling_ + 0.0001)
            border_sampling               = len_border / ( param_border_subsampling_sill/param_border_subsampling_ * ( 1 + tt - np.sqrt( (tt +1)**2  -2*param_border_subsampling_*tt )  )   ) 
            
            temp_position_border          = temp_position_border[::np.round(border_sampling).astype(int)]
            temp_dtm_border_flood_smooth  = temp_dtm_border_flood_smooth[::np.round(border_sampling).astype(int)]
            
            len_border                    = len(temp_position_border)
            # 二次取樣的機制
            # tt: 當前邊界像素數量與設定的最大邊界像素數量的比值
            # param_border_subsampling_: 工程手法由2-param_border_subsampling轉換而來，避免為0，確保不要出現0，至少0.0001
            # border_sampling: 每隔多少像素點取樣一次，當tt很小，border_sampling接近1(不取樣)
            # 當tt很大，border_sampling接近 len_border / param_border_subsampling_sill (大約每 param_border_subsampling_sill 個像素取樣一次)
            
            # BUILD A CKDTREE FOR EACH SET OF POINTS
            tree_BORDER = cKDTree(temp_position_border)
            #是一種資料結構，可以高效地進行空間查詢，特別適用於最近鄰搜尋
            
            param_workers_cKDTree =  8
            distances, indices = tree_BORDER.query(temp_position_flood, k=param_max_number_neighbors,workers = param_workers_cKDTree)
            # 查詢 temp_position_flood 中每個點在 temp_position_border 中的 k 個最近鄰
            distances = distances.astype('float32')
            distances[distances==0]  = 1
            # 如果恰巧若在邊界點上，距離會是0，會導致權重無限大，因此將距離為0的點設為1(避免無限大)
            indices   = indices.astype('uint32')


            if distances.ndim == 1:
                distances = np.expand_dims(distances, axis=1)
                indices   = np.expand_dims(indices, axis=1)
            # 如果只有一個鄰居，將距離和索引陣列擴展為二維陣列，以便後續處理

            if param_max_number_neighbors > len_border:
                indices   = indices  [:, :len_border]
                distances = distances[:, :len_border]
            # 如果設定的最大鄰居數量超過邊界像素數量，則將鄰居數量限制為實際的邊界像素數量，避免無效索引

            
            #####################
            ## CHOOSE THE WEIGHTS
            #####################
                
            # inverse distance weighting IDW
            weights  =   1 / (distances ** param_inverse_dist_exp).astype('float32')


            #####################################
            ## choose WL estimation method A vs B
            #####################################

            if param_WL_estimation_method == 'method_A':
            #METHOD A : water level is interpoleated using IDW or EDW weights
                dtm_temp_statistic    =  (np.sum((temp_dtm_border_flood_smooth[indices] * weights).astype('float32'), axis = 1 )  / np.sum( weights , axis = 1 ) ).astype('float32')
            # 透過距離的反比權重來插值計算水位
                
            elif param_WL_estimation_method == 'method_B':
                #METHOD B : water level is a distence-weighted QUANTILE of all closest dtm neighboring cells 
                dtm_temp_statistic    =   weighted_quantile(temp_dtm_border_flood_smooth[indices], param_border_quantile, weights  =  weights)
            # 使用加權分位數來估計水位，這種方法可以更好地處理邊界值和異常值                
                    
            z_water_level[temp_position_flood[:,0], temp_position_flood[:,1]]    =    dtm_temp_statistic 
            #plt.imshow(z_water_level, interpolation = 'none', vmin=50, vmax=100)
    
    
    

    ###############
    #################
    ##FLOOD EXPANSION
    ###################
    #####################

    z_water_level_augmented = np.copy(z_water_level)
    # 如果想要洪水擴展到排除遮罩外的區域，或者有提供排除遮罩檔案，則進行洪水擴展
    if os.path.isfile(exclusion_path) or param_spread_outside_exclusion_mask : 
        print('Augmenting flooded areas...')
        # 增強洪水區域
        if param_spread_outside_exclusion_mask == True :
            exclusion = np.full_like(exclusion, 1)
        # 擴展 = True 將排除區填滿1
        param_connectivity                 =  8         # <<INPUT connectivity used to spread flooded areas (4 or 8)
        # 連通性設為8 (對角也要)
        z_water_level_masked_compressed         =   np.ma.masked_array(z_water_level, flood_border_inner==0).compressed() 
        z_water_level_masked_compressed_initial =   np.ma.masked_array(z_water_level, flood_border_inner==0).compressed()
        # 這裡純粹是當前水位的副本
        row_flood_border                        =   np.ma.masked_array(ROW,           flood_border_inner==0).compressed()
        col_flood_border                        =   np.ma.masked_array(COL,           flood_border_inner==0).compressed()
        flood_labels_border                     =   np.ma.masked_array(z_labels,      flood_border_inner==0).compressed()
        # 選擇與 flood_border_inner==0 不符的元素
        # np.ma.masked_array(data,mask) 在這裡就是將 flood_border_inner != 0 的位置在 data 中遮罩起來(被忽略)
        # .compressed() 會將該陣列"沒有被遮罩"的元素提出來組合成新的一維陣列
        
        FLOOD = np.array ( [flood_labels_border,  row_flood_border,  col_flood_border,  z_water_level_masked_compressed, z_water_level_masked_compressed_initial ]  ).T
        # 將Flood建立成 5 X N的陣列的轉置 N X 5

        #SORTS BASED ON DECREASING WATER ELEVATION
        FLOOD_sort_descending= np.flipud(FLOOD[FLOOD[:, 3].argsort()])
        # 根據第四欄( z_water_level_masked_compressed ) 欄 來排序
        # .argsort() 會回傳對應的索引被排序後的新索引 這裡是升序 ascending
        # Flood[] 照著回傳索引排順序
        # np.flipud() 會將二維陣列沿著垂直軸(上到下)翻轉 所以會得到 降序 descending  水位高的會在最上面 

        #SIZE OF INITIAL FLOODED AREAS IN M2
        dim_flooded_area = z_stats[:,4] * L**2
        dim_flooded_area[dim_flooded_area<0]=0  #  prevents potential negative values caused by overflow 
        
        
        ########################
        # MAX EXPANSION DISTANCE  
        ########################
       
        ##EXPONENTIAL PARAMETRIZATION OF MAX PROPAGATION DISTANCE
        threshold_distance = param_max_propagation_distance * 1000   *  ( 1 -   pow(2,   -  dim_flooded_area   / (param_distance_range * 1e6  )  )        )
        # 最大傳播距離*1000*(1-2^(-淹水區域面積/距離範圍))
        #param_distance_range "A_1/2" 距離範圍 [0,100] (km^2) 與D_max(最大傳播距離)共同作用，定義洪水傳播距離與淹水區域大小的關係。
        # -->設為10，代表面積為10平方公里的淹水區域，其水流最多傳播到Dmax的一半

        #avoids singularities in case no flood expansion is set
        threshold_distance = threshold_distance + 1
        # 避免除以0出現錯誤

        FLOOD_sort_descending_list= list (FLOOD_sort_descending)
        # 建立列表

        z_water_level_augmented = np.copy(z_water_level)
        # 動態變化可更新的水位

        z_labels_augmented      = np.copy(z_labels)
        temp_dist_from_border   = np.zeros(z_labels.shape, dtype = 'float16')
        # 記錄新的淹水點與新擴散區域的標籤
        # 建立與標籤形狀相同的全0陣列
        
        i=0
        i_counter  =  len ( FLOOD_sort_descending_list )
        i_stop     =  len ( FLOOD_sort_descending_list )
        total_processed = 0
        
        wl_frames_expand = []
        snap_every = 2000         # 每處理 2000 個像素存一幀（可調）
        preview_max_frames = 120  # 最多存多少幀，避免 GIF 過大

        def render_frame(arr2d, mask=None, title=None):
            """回傳一張 numpy uint8 圖片（先渲染成 PNG，再讀回）"""
            import io
            fig = plt.figure(figsize=(5,5))
            im = plt.imshow(arr2d, interpolation='nearest')  # 不指定色盤，保持預設
            plt.axis('off')
            if title: plt.title(title)
            if mask is not None:
                m = np.ma.masked_where(~mask.astype(bool), mask)
                plt.imshow(m, alpha=0.15)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            img = imageio.imread(buf)
            buf.close()
            return img

        # 先存初始一幀（外擴前的 z_water_level_augmented）
        wl_frames_expand.append(
            render_frame(z_water_level_augmented.copy(), mask=(flood>0), title='Start')
        )

        # 跑完整個迴圈
        while i < i_counter: 
            
            # 當執行到列表長度(最後一組)
            if i == i_stop:
                FLOOD_sort_descending = np.vstack(FLOOD_sort_descending_list)
                FLOOD_sort_descending = FLOOD_sort_descending[i_stop: ,:]
                FLOOD_sort_descending = np.flipud(FLOOD_sort_descending[FLOOD_sort_descending[:, 3].argsort()]) 
                FLOOD_sort_descending_list= list (FLOOD_sort_descending)
                i=0
                i_stop    = len(FLOOD_sort_descending)
                i_counter = len(FLOOD_sort_descending) 
            # 將所有列表元素垂直堆疊
            # 在最後一組後新增單元
            # 根據第四欄( z_water_level_masked_compressed ) 欄 來排序
            # 上下翻轉後仍然是由高水位排到低水位
            # 轉成list
            # 重新紀錄參數
        
        
            vicini =  ij_neighbors(FLOOD_sort_descending[i,1], FLOOD_sort_descending[i,2] , n_row, n_col, param_connectivity)  
            # 檢查鄰居數量與相連   ij_neighbors(目標像素i,目標像素j,網格總列數,網格總欄數,連接性)
            # row_flood_border,  col_flood_border  為 target cell 的 row column 

            label     = FLOOD_sort_descending[i,0].astype('uint32')
            
            initial_wl= FLOOD_sort_descending[i,4]  # 初始水位是用 z_water_level_masked_compressed_initial
        
        
    
            temp_= ( L +  temp_dist_from_border[FLOOD_sort_descending[i,1].astype('uint32'), FLOOD_sort_descending[i,2].astype('uint32')]  ) / threshold_distance[label]
            # 後續擴散用的 距離比例因子  temp_ = (網格大小+已淹水區邊界到當前處理單元格i的累計距離)/最大傳播距離閾值

            wl = (  initial_wl  -  ( initial_wl  - dtm[vicini[:,0],vicini[:,1]]   )  * temp_  )     * np.heaviside(1-temp_ , 0 )   +    dtm[vicini[:,0],vicini[:,1]]  *     (  1- np.heaviside(1-temp_ , 0 )    )
            # 水位計算公式
            # 當temp_接近0時，wl會接近 initial_wl，而 temp_接近1(接近threshold_distance)時，wl會接近 dtm[vicini]
            # 這裡使用heviside 如果 temp_< 1 ， H = 1，反之 temp_ > 1 ， H = 0 為了避免 temp_出現不合理的情況
            # 如果出現 H = 0，則使用DTM作為水位高度 wl = DTM
            # 使用dtm[vicini] 是一個 N X 2 的陣列，N為所設定的連接性
            # 當N為8，一次會算出八個位置(鄰居)的wl

            wl[wl>FLOOD_sort_descending[i,3]] = FLOOD_sort_descending[i,3]
            
            r = vicini[:, 0]
            c = vicini[:, 1]
            domain_mask = ((exclusion[r, c] > 0) |(obswater[r, c] > 0) |(permanent_water[r, c] > 0))
            elevation_mask = (dtm[r, c] < (wl - 0.01))
            unassigned_mask = np.isnan(z_water_level_augmented[r, c])
            if uncertainty_on:
                if uncertainty_gate_mode == "semantic_high_uncertainty":
                    uncertainty_mask = semantic_candidate_mask[r, c]
                else:
                    unc = uncertainty[r, c]
                    uncertainty_mask = (
                        np.isfinite(unc)
                        & (unc < uncertainty_threshold)
                    )
            else:
                uncertainty_mask = np.ones_like(domain_mask, dtype=bool)

            # UNCERTAINTY-FLEXTH CHANGE U4: the optical gate is one necessary
            # condition; FLEXTH elevation, water level and topology still decide.
            condition = (domain_mask & elevation_mask & unassigned_mask & uncertainty_mask)
            
            if debug_uncertainty:
                total_neighbors = len(r)

                n_domain = np.sum(domain_mask)
                n_domain_elev = np.sum(domain_mask & elevation_mask)
                n_domain_elev_unassigned = np.sum(domain_mask & elevation_mask & unassigned_mask)
                n_final = np.sum(condition)

                if uncertainty_on:
                    n_unc_pass = np.sum(
                        domain_mask &
                        elevation_mask &
                        unassigned_mask &
                        uncertainty_mask
                    )

                    n_unc_blocked = np.sum(
                        domain_mask &
                        elevation_mask &
                        unassigned_mask &
                        (~uncertainty_mask)
                    )
                else:
                    n_unc_pass = n_final
                    n_unc_blocked = 0

                print("------ DEBUG FLOOD CONDITION ------")
                print(f"gate mode: {uncertainty_gate_mode}")
                print(f"total neighbors: {total_neighbors}")
                print(f"domain pass: {n_domain}")
                print(f"domain + elevation pass: {n_domain_elev}")
                print(f"domain + elevation + unassigned pass: {n_domain_elev_unassigned}")
                print(f"uncertainty pass: {n_unc_pass}")
                print(f"uncertainty blocked: {n_unc_blocked}")
                print(f"final condition pass: {n_final}")
                print("-----------------------------------")
            #condition =  (   
                
                         #np.heaviside( (exclusion[vicini[:,0],vicini[:,1]] > 0)  + (obswater[vicini[:,0],vicini[:,1]] > 0) + (permanent_water[vicini[:,0],vicini[:,1]] > 0)   , 0).astype('bool')          # 1. must be in the exlusion or permanent water layer
    
                       #* ( dtm[vicini[:,0],vicini[:,1]] <  (wl-0.01) ) # FLOOD_sort_descending[i,3] #  z_flooded_areas_augmented[i][2]                   # 2. dtm must be lower than the corresponding water level    
                     
                       #* np.heaviside(  np.isnan( z_water_level_augmented[vicini[:,0],vicini[:,1]])         , 0  ).astype('bool')                        # 3. must be nodata (i.e. not already assigned)
                       #* (
                            #((uncertainty[vicini[:,0],vicini[:,1]] <= uncertainty_threshold)
                                #& np.isfinite(uncertainty[vicini[:,0],vicini[:,1]]))| (not uncertainty_on) # 4. uncertainty must be below threshold
                         #))
            # 組合三個獨立的布林條件
            # 1. 必須要是潛在的淹水區域 (排除區域、永久水體、觀測水體)
            # 2. 地形高度必須比計算出的水位低
            # 3. 必須要是個 no data 的區域 並非flood extent = 0
        
           
            temp_dist_from_border[vicini[:,0][condition],vicini[:,1][condition]]    = L * np.array([1.414, 1, 1.414, 1, 1.414, 1, 1.414, 1])[condition]     + temp_dist_from_border[FLOOD_sort_descending[i,1].astype('uint32'), FLOOD_sort_descending[i,2].astype('uint32')]
            # 新的距離 = 步長距離(單位邊長到鄰居單元格所需走的額外距離) + 中心點舊距離

            z_water_level_augmented[vicini[:,0][condition],vicini[:,1][condition]]  =  wl[condition]   
            # 將新淹水區域寫入水位
            
            FLOOD_sort_descending_list.append  (  np.array([ label*(condition[condition]) ,  vicini[:,0][condition],  vicini[:,1][condition]  ,  wl[condition] , initial_wl*(condition[condition])  ]).T  ) 
            # 新的淹水區被加入列表

            z_labels_augmented[vicini[:,0][condition],vicini[:,1][condition]]  =  label 
            
            i_counter+= np.sum(condition)
            # 計算有多少新的淹水區域 使迴圈條件 i < i_counter成立 動態擴展            
                                 
            i+=1 
            
            #print(i)
            total_processed +=1

            if (total_processed % snap_every) == 0 and len(wl_frames_expand) < preview_max_frames:
                wl_frames_expand.append(
                    render_frame(z_water_level_augmented, mask=(flood>0),
                                title=f'Expanded: {total_processed:,} px')
                )
        # 補最後一幀
        wl_frames_expand.append(
        render_frame(z_water_level_augmented, mask=(flood>0), title='Final')
        )
        # 儲存 GIF
        out_gif = Path(output_dir) / 'flood_expansion.gif'
        imageio.mimsave(out_gif, wl_frames_expand, fps=6)   # 6 張/秒，可調
        print('Saved GIF:', out_gif)
    
        ##############################
        ## SMOOTHING EXPANDED WL ####
        ##############################
        print('Smoothing... \n')
               
        param_number_smoothings      = 20   #<<<<<<  INPUT - how many smoothing iterations 
        # 要做幾次平滑
        
        z_water_level_augmented_initial = np.copy(z_water_level_augmented)
        z_water_level_augmented[ np.isnan(z_water_level_augmented_initial) ] = dtm[  np.isnan(z_water_level_augmented_initial)  ]
        # 水位為Nan的地方先用DTM補起來

        target_i, target_j  =  np.where( (~np.isnan(z_water_level_augmented_initial)) & (flood == 0) )
        target_i = target_i.astype(np.int32) ; target_j = target_j.astype(np.int32)
        # 找到有水位資料的格點且不是洪水區域的資料
        # ~代表NOT
        
        connectivity = 24      #<<<<<<  INPUT - size of the smoothing kernel (24 is a 5x5 box kernel, 8 is a 3x3 box kernel)
        
        neighbors_i = np.zeros((target_i.shape[0], connectivity)).astype(np.uint32)
        neighbors_j = np.zeros((target_i.shape[0], connectivity)).astype(np.uint32)
        # 建立兩個大小為(目標格點數量，鄰居數量)的全0陣列
          
        for i in range(target_i.shape[0]):
            
            neighbors = ij_neighbors(target_i[i], target_j[i], n_row, n_col, connectivity)
            neighbors_i[i, :] = neighbors[:, 0]
            neighbors_j[i, :] = neighbors[:, 1]
            # 把鄰居的row col放進陣列
            # neighbors 是一個 (N, 2) 的陣列 N為連接性
                    
        
        mask_1 = np.where(~np.isnan(z_water_level))
        # 找到原本就有水位資料的格點
        
        for v in range(param_number_smoothings): # 這裡做20次
            z_water_level_augmented[target_i,target_j] = np.mean(z_water_level_augmented[neighbors_i,neighbors_j], axis = 1 )
            z_water_level_augmented[ mask_1]           = z_water_level[  mask_1  ]
        # 對目標格點做平滑
        # 目標格點的水位 = 鄰居格點水位的平均值
        # 但原本就有水位資料的格點，還是維持原本的水位不變
        z_water_level_augmented[ np.isnan(z_water_level_augmented_initial) ] = np.nan
        # 最後再把原本水位為Nan的地方設回Nan
        

    else:
        z_water_level_augmented = np.copy(z_water_level)
        # 如果不需要洪水擴展，直接使用初始水位
    

    ###############
    # WATER DEPTH #
    ###############
    WD                        = ( ( z_water_level_augmented - dtm ) * 100  ).astype('float32')   
    # 水深以公分為單位
    WD[WD<0] = 0 
    WD[WD > 0]                = WD[WD > 0] + param_WD_star
    # 將水深大於0的部分加上最小水深參數(以公分為單位)
    WD[(WD==0) & (flood>0)  ] = param_WD_star
    # 將初始淹水區域中水深為0的部分設為最小水深參數(以公分為單位)
    
    #dummy water depth assigned to permanent water bodies
    WD[permanent_water==1] = 9999 
    WD[np.isnan(WD)] = 0
    
    
    
    ############
    # WL LEVEL #
    ############
    WL          = ( z_water_level_augmented   ).astype('float32')   
    WL[WD > 0]  = WL[WD > 0] + param_WD_star/100
    # 將水深大於0的部分加上最小水深參數(以公尺為單位)
    WL[WD==0 ]  = np.nan
    
    #DUMMY WATER LEVEL ASSIGNED TO PERMANENT WATER BODIES
    WL[permanent_water==1] = 9999 
    

    ########
    ##SAVING
    ########
    
    print('Saving... \n')
    
    prefix  = os.path.splitext(os.path.basename(flood_path))[0]
    
    if param_output_map == "WD" or  param_output_map == "WL_WD" :
    
        with rasterio.open(
            output_dir / f'WD_{prefix}_{param_WL_estimation_method}_Smax_{param_threshold_slope}_Nmax_{param_max_number_neighbors}_a_{param_inverse_dist_exp}_Sbs_{param_border_subsampling}_Dmax_{param_max_propagation_distance}_A12_{param_distance_range}_gaps_{param_size_gaps_close}_unrestricted_{param_spread_outside_exclusion_mask}.tif',
            mode="w",
            driver="GTiff",
            compress='deflate',
            height=n_row,
            width=n_col,
            count=1,
            dtype=rasterio.uint16,
            crs=crs,
            transform=transform,
            nodata= 0
            ) as water_depth:
                water_depth.write(WD, 1)
                
                
    if param_output_map == "WL" or  param_output_map == "WL_WD" :
    
        with rasterio.open(
            output_dir / f'WL_{prefix}_{param_WL_estimation_method}_Smax_{param_threshold_slope}_Nmax_{param_max_number_neighbors}_a_{param_inverse_dist_exp}_Sbs_{param_border_subsampling}_Dmax_{param_max_propagation_distance}_A12_{param_distance_range}_gaps_{param_size_gaps_close}_unrestricted_{param_spread_outside_exclusion_mask}.tif',
            mode="w",
            driver="GTiff",
            compress='deflate',
            height=n_row,
            width=n_col,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            ) as water_level:
                water_level.write(WL, 1)
    
   
    
    print('Finish ! \n')
      
        
########        
## RUN ! 
########     
  
if __name__ == '__main__':
    
    debug_uncertainty = False  # True prints per-pixel gate counts (very verbose)
    start = time.time()  
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    if param_tiling == True: 
       
        #TILES ALL THE INPUTS IF PARAM_TILE_INPUTS IS TRUE 
        if param_tile_inputs == True:             
            print('Tiling the input rasters...')
            
            tiling(input_dir / 'flood.tif' , input_dir, param_tile_size)
            
            tiling(input_dir / 'dtm.tif'   , input_dir, param_tile_size)
            
            if os.path.isfile(input_dir / 'exclusion.tif'):
                tiling(input_dir / 'exclusion.tif' , input_dir, param_tile_size)
                
            if os.path.isfile(input_dir / 'obswater.tif'):
                tiling(input_dir / 'obswater.tif' , input_dir, param_tile_size)
                
            if os.path.isfile(input_dir / 'permanent_water.tif'):
                tiling(input_dir / 'permanent_water.tif' , input_dir, param_tile_size)

            if uncertainty_on:
                tiling(input_dir / 'uncertainty.tif', input_dir, param_tile_size)
                if uncertainty_gate_mode == "semantic_high_uncertainty":
                    tiling(input_dir / 'classification.tif', input_dir, param_tile_size)
                    tiling(input_dir / 'water_probability.tif', input_dir, param_tile_size)
            

    
        input_list_flood = glob.glob( str(input_dir/"**/*flood_tile*.tif")  , recursive=True)
       
        for index, flood_path in enumerate(input_list_flood):
            
            print(f'Processing tile {index+1} out of {len(input_list_flood)}')
            
            dtm_path             =  flood_path.replace("flood_tile", "dtm_tile") 
            exclusion_path       =  flood_path.replace("flood_tile", "exclusion_tile") 
            obswater_path        =  flood_path.replace("flood_tile", "obswater_tile")  
            permanent_water_path =  flood_path.replace("flood_tile", "permanent_water_tile") 
            uncertainty_path     =  flood_path.replace("flood_tile", "uncertainty_tile")
            classification_path  =  flood_path.replace("flood_tile", "classification_tile")
            water_probability_path = flood_path.replace("flood_tile", "water_probability_tile")
    
            # UNCERTAINTY-FLEXTH CHANGE U5: pass aligned semantic inputs to each tile.
            flood_processing(flood_path           = flood_path ,
                             dtm_path             = dtm_path  ,
                             uncertainty_path     = uncertainty_path ,
                             classification_path  = classification_path ,
                             water_probability_path = water_probability_path ,
                             exclusion_path       = exclusion_path ,
                             obswater_path        = obswater_path , 
                             permanent_water_path = permanent_water_path )   

       
        
        
    else:
        flood_processing(flood_path           = input_dir / 'flood.tif' ,
                         dtm_path             = input_dir / 'dtm.tif'   ,
                         uncertainty_path     = input_dir / 'uncertainty.tif' ,
                         classification_path  = input_dir / 'classification.tif' ,
                         water_probability_path = input_dir / 'water_probability.tif' ,
                         exclusion_path       = input_dir / 'exclusion.tif' ,
                         obswater_path        = input_dir / 'obswater.tif' , 
                         permanent_water_path = input_dir / 'permanent_water.tif' )        
    
    
    #MERGE WD AND WL TILES
    if param_tiling == True and param_merge_tiles == True:
        
        tiles_to_merge_WD = glob.glob( str(output_dir/"*WD_flood_tile*.tif")  , recursive=True)
        tiles_to_merge_WL = glob.glob( str(output_dir/"*WL_flood_tile*.tif")  , recursive=True)
    
        inputs = [rasterio.open(f) for f in tiles_to_merge_WD]
        merged_image, out_trans = rasterio.merge.merge(inputs)

        with rasterio.open(output_dir / f'WD_merge_{param_WL_estimation_method}_Smax_{param_threshold_slope}_Nmax_{param_max_number_neighbors}_a_{param_inverse_dist_exp}_Sbs_{param_border_subsampling}_Dmax_{param_max_propagation_distance}_A12_{param_distance_range}_gaps_{param_size_gaps_close}_unrestricted_{param_spread_outside_exclusion_mask}.tif', 
                           'w', 
                           driver='GTiff',
                           height=merged_image.shape[1], 
                           width=merged_image.shape[2], 
                           count=1, 
                           dtype=merged_image.dtype, 
                           crs=inputs[0].crs, 
                           nodata = inputs[0].nodata,
                           transform=inputs[0].transform,
                           compress='deflate')  as dst:
            dst.write(merged_image[0,:,:], 1)
        
        for input in inputs:
            input.close()
        
        
    
        inputs = [rasterio.open(f) for f in tiles_to_merge_WL]
        merged_image, out_trans = rasterio.merge.merge(inputs)

        with rasterio.open(output_dir / f'WL_merge_{param_WL_estimation_method}_Smax_{param_threshold_slope}_Nmax_{param_max_number_neighbors}_a_{param_inverse_dist_exp}_Sbs_{param_border_subsampling}_Dmax_{param_max_propagation_distance}_A12_{param_distance_range}_gaps_{param_size_gaps_close}_unrestricted_{param_spread_outside_exclusion_mask}.tif', 
                           'w', 
                           driver='GTiff',
                           height=merged_image.shape[1], 
                           width=merged_image.shape[2], 
                           count=1, 
                           dtype=merged_image.dtype, 
                           crs=inputs[0].crs, 
                           nodata = inputs[0].nodata,
                           transform=inputs[0].transform,
                           compress='deflate')  as dst:
            dst.write(merged_image[0,:,:], 1)
        
        for input in inputs:
            input.close()
        

    # COMPUTE TOTAL PROCESSING TIME
    end = time.time()
    print(f'Total processing time:{int((end - start)/60)} minutes')
