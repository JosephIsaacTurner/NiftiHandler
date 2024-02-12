import nibabel as nib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas() 

# Get the directory in which the current script is located
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Construct the path to the 'atlases' directory
atlases_dir = os.path.join(parent_dir, 'atlas_data')

class Atlas:
    """
    A class for handling and accessing data from neuroimaging atlases.

    This class is designed to load and manage an atlas used in neuroimaging, typically consisting of
    a Nifti file (.nii or .nii.gz) representing brain regions and an accompanying CSV file that maps
    numerical values in the Nifti file to named regions.

    Attributes:
        csv_key_path (str): Path to the CSV file containing the key for the atlas.
        key (pd.DataFrame): DataFrame loaded from the CSV file. It maps values in the Nifti file to named regions.
        atlas (nib.Nifti1Image): Nifti image loaded from the Nifti file.
        labels (list): List of region names extracted from the CSV key.

    Args:
        filepath (str): Path to the .nii/.nii.gz file of the atlas or to the corresponding .csv file.
        csv_key_path (str, optional): Optional path to the .csv file containing the key for the atlas.
                                      If not provided, the class attempts to find a .csv file matching
                                      the root name of the provided Nifti file.
    """

    def __init__(self, filepath=os.path.join(atlases_dir, "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz"), csv_key_path=None):
        """
        Initializes the Atlas class by loading the atlas Nifti file and the corresponding CSV key.
        """
        
        # Determine the root file name without extension
        root_filepath = os.path.splitext(filepath)[0]
        if root_filepath.endswith('.nii'):
            root_filepath = root_filepath[:-4]

        # Set up file paths for .csv and .nii/.nii.gz files
        if not csv_key_path:
            csv_key_path = root_filepath + ".csv"

        nii_file_path = root_filepath + ".nii.gz"
        if not os.path.exists(nii_file_path):
            nii_file_path = root_filepath + ".nii"
            if not os.path.exists(nii_file_path):
                raise FileNotFoundError(f"Could not find a .nii or .nii.gz file for {root_filepath}, trying {nii_file_path}.")

        if not os.path.exists(csv_key_path):
            raise FileNotFoundError(f"Could not find the csv file {csv_key_path}.")

        # Load the atlas and the key
        self.csv_key_path = csv_key_path
        self.key = pd.read_csv(self.csv_key_path)
        self.atlas = nib.load(nii_file_path)
        self.labels = self.key['name'].tolist()
    
    def name_at_index(self, index=[48, 94, 35]):
        """
        Retrieves the name of the brain region corresponding to a given index in the atlas.

        This method looks up the value at the specified index in the Nifti file and uses the CSV key
        to return the name of the region associated with that value.

        Args:
            index (list): A list of three integers representing the x, y, z coordinates of the index (in voxel space).

        Returns:
            str or list: The name of the region at the given index. If multiple regions are found at the index,
                         returns a list of names. Returns "No matching region found" if no region matches the index.
        """
        
        value_at_index = self.atlas.get_fdata()[tuple(index)]
        matched_row = self.key[self.key['value'] == value_at_index]
        if len(matched_row) == 1:
            return matched_row['name'].iloc[0]
        elif len(matched_row) > 1:
            return matched_row['name'].tolist()
        else:
            return "No matching region found"

class AtlasLabeler:
    """
    A class to label Nifti volumes based on an atlas.

    This class is designed to take a Nifti volume and an atlas, and label the volume
    based on the regions defined in the atlas. The labeling considers the intensity
    thresholds specified for the Nifti volume.

    Attributes:
        atlas (Atlas): An Atlas object containing the atlas data and methods.
        labels (list): List of region names from the atlas.
        nifti (nib.nifti1.Nifti1Image): The Nifti image to be labeled.
        volume_data (numpy.ndarray): Data from the Nifti image.
        labeled_data (pd.DataFrame or None): DataFrame containing labeled voxels. Populated after label_volume is called.
        voxel_counts (dict or None): Dictionary of counts of labeled voxels per region. Populated after label_volume is called.
        unique_labels (list or None): List of unique labels assigned. Populated after label_volume is called.
        min_threshold (int): Minimum intensity threshold for considering a voxel in the Nifti volume.
        max_threshold (int): Maximum intensity threshold for considering a voxel in the Nifti volume.

    Args:
        nifti (str or nib.nifti1.Nifti1Image): The Nifti volume to be labeled, either as a file path or Nifti image object.
        atlas (Atlas or str): An Atlas object or the file path to the atlas (.nii.gz file).
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
    """

    def __init__(self, nifti, atlas=os.path.join(atlases_dir,"HarvardOxford-cort-maxprob-thr0-2mm.nii.gz"), min_threshold=1, max_threshold=1):
        """
        Initializes the AtlasLabeler class with a Nifti volume, an atlas, and intensity thresholds.
        """

        if type(atlas) != Atlas:
            # print(f"Looking for atlas {atlas}...")
            try:
                atlas = Atlas(atlas)
            except:
                print(f"""Could not find atlas {atlas}. Make sure the path is correct and try again. 
                      \n Example: atlases/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz""")
                return
        self.atlas = atlas
        self.labels = atlas.labels

        if type(nifti) != nib.nifti1.Nifti1Image:
            # print(f"Looking for nifti {nifti}...")
            try:
                nifti = nib.load(nifti)
            except:
                print(f"""Could not find nifti {nifti}. Make sure the path is correct and try again. 
                      \n Example: volumes/subject_lesion_mask.nii.gz""")
                return
        self.nifti = nifti
        self.volume_data = nifti.get_fdata()
        self.labeled_data = None
        self.voxel_counts = None
        self.unique_labels = None
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def label_volume(self):
        """
        Labels the Nifti volume using the regions defined in the atlas.

        This method iterates through each voxel in the Nifti volume, labeling it based on the
        overlapping regions in the atlas masks. Only voxels with intensity values within the
        specified threshold range are considered for labeling.

        The method updates the labeled_data, voxel_counts, and unique_labels attributes of the
        class with the results of the labeling process.

        Returns:
            AtlasLabeler: The instance itself with updated labeled_data, voxel_counts, and unique_labels.
        """        
        if self.atlas.atlas.shape != self.nifti.shape:
            print(f"The shape of the atlas ({self.atlas.atlas.shape}) and the nifti volume ({({self.nifti.shape})}) do not match. Please provide a nifti volume with the same shape as the atlas.")
            return

        # Find indices where the volume data is within the specified threshold range
        masked_indices = np.where(np.logical_and(self.volume_data >= self.min_threshold, 
                                         self.volume_data <= self.max_threshold))
        # Prepare a dictionary to store the results
        results = {'index': [], 'atlas_label': []}

        # Iterate over the masked indices
        for i, j, k in zip(*masked_indices):
            label = self.atlas.name_at_index([i, j, k])

            # Store the results
            results['index'].append((i, j, k))
            results['atlas_label'].append(label)

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        self.labeled_data = results_df
        self.voxel_counts = results_df['atlas_label'].value_counts().to_dict()
        self.unique_labels = results_df['atlas_label'].unique().tolist()
        return self

class MultiAtlasLabeler:
    """
    A class for labeling a Nifti volume using multiple atlases.

    This class enables the labeling of a Nifti volume by applying multiple atlas definitions
    with specified intensity thresholds. It consolidates the labeling results from all the
    atlases, providing comprehensive labeling based on multiple sources.

    Attributes:
        atlas_list (pd.DataFrame): A DataFrame containing paths to atlases and their corresponding loaded Atlas objects.
        labels (list): A list of unique labels across all atlases.
        nifti_path (str or nib.Nifti1Image): The file path or Nifti image object of the volume to be labeled.
        voxel_counts (dict or None): Counts of labeled voxels per region after label_volume is called (aggregated across all atlases).
        unique_labels (list or None): Unique labels assigned to the volume after label_volume is called (aggregated across all atlases).
        min_threshold (int): Minimum intensity threshold for considering a voxel in the labeling process.
        max_threshold (int): Maximum intensity threshold for considering a voxel in the labeling process.

    Args:
        nifti_path (str or nib.Nifti1Image): The file path or Nifti image object of the volume to be labeled.
        atlas_list (str or list or dict): Path to a CSV file containing atlas paths, a list of atlas paths, or a dictionary with atlas paths. The CSV file or dictionary should have a column or key named 'atlas_path'.
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
    """  
    def __init__(self, nifti_path, atlas_list=os.path.join(atlases_dir,'harvoxf_atlas_list.csv'), min_threshold=1, max_threshold=1):
        # Store the thresholds
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Load atlas_list from a CSV file
        if isinstance(atlas_list, str):
            try:
                atlas_list = pd.read_csv(atlas_list)
                atlas_list['atlas'] = atlas_list['atlas_path'].apply(lambda x: Atlas(x))
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find csv {atlas_list}. Make sure the path is correct and try again.")

        # Handling atlas_list if it's a list or a dict
        elif isinstance(atlas_list, (list, dict)):
            processed_atlas_list = [Atlas(item) if not isinstance(item, Atlas) else item for item in atlas_list]
            atlas_list = pd.DataFrame({'atlas': processed_atlas_list})

        else:
            raise ValueError("Invalid type for atlas_list. Must be a string, list, or dictionary.")

        # Load Nifti file
        if not isinstance(nifti_path, nib.nifti1.Nifti1Image):
            try:
                self.nifti_path = nib.load(nifti_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find nifti {nifti_path}. Make sure the path is correct and try again.")
        else:
            self.nifti_path = nifti_path

        consolidated_unique_labels = []
        for atlas in atlas_list['atlas']:
            labels = atlas.labels
            consolidated_unique_labels.extend(labels)
        
        self.labels = list(set(consolidated_unique_labels))

        self.atlas_list = atlas_list
        self.voxel_counts = None
        self.unique_labels = None
    
    def label_volume(self):
        """
        Labels the Nifti volume using the atlases provided in the atlas list.

        This method applies each atlas in the atlas list to label the Nifti volume. It then consolidates
        the labeling results from all the atlases, providing a comprehensive set of labels for the volume.

        The method updates the voxel_counts and unique_labels attributes of the class with the consolidated results.

        Returns:
            MultiAtlasLabeler: The instance itself with updated voxel_counts and unique_labels.
        """

        def safe_label_volume(nifti_path, atlas, min_threshold, max_threshold):
            try:
                labeler = AtlasLabeler(nifti_path, atlas, min_threshold, max_threshold)
                labeler.label_volume()
                return labeler
            except Exception as e:
                print(f"Error processing {atlas}: {e}")
                return None

        # List to hold the results for each atlas
        results_list = []

        # Iterate through the atlas list and label the volume for each atlas
        for _, row in self.atlas_list.iterrows():
            labeler = safe_label_volume(self.nifti_path, row['atlas'], self.min_threshold, self.max_threshold)
            if labeler and labeler.labeled_data is not None:
                voxel_counts = labeler.voxel_counts
                unique_labels = labeler.unique_labels
                results_list.append({'voxel_counts': voxel_counts, 'unique_labels': unique_labels})

        # Convert the list of results to a DataFrame using pd.concat
        if results_list:
            # Aggregate voxel counts
            combined_voxel_counts = {}
            for result in results_list:
                for label, count in result['voxel_counts'].items():
                    combined_voxel_counts[label] = combined_voxel_counts.get(label, 0) + count
            self.voxel_counts = combined_voxel_counts

            # Aggregate unique labels
            all_labels = set()
            for result in results_list:
                all_labels.update(result['unique_labels'])
            self.unique_labels = list(all_labels)
        else:
            print("No valid results were generated from the atlases.")
            self.voxel_counts = {}
            self.unique_labels = []

        return self

class CustomAtlas:
    """
    A class for managing an atlas used in neuroimaging analysis.

    This class handles the loading and management of an atlas, which includes a set
    of regions (each with a corresponding mask) used for analyzing or labeling Nifti volumes.
    
    Attributes:
        atlas_df (pd.DataFrame): A DataFrame representing the atlas. Each row corresponds to a region, with columns for region name and mask file path.
        labels (list): A list of region names present in the atlas.

    Args:
        atlas_path (str or dict or pd.DataFrame): The path to the CSV file representing the atlas, 
                                                  a dictionary, or a DataFrame with atlas information.
                                                  The CSV or DataFrame should have columns for 'region_name' 
                                                  and 'mask_path'. If using a dictionary, the keys should be
                                                  'region_name' and 'mask_path'.
    """

    def __init__(self, atlas_path=os.path.join(atlases_dir, "joseph_custom_atlas.csv")):
        if type(atlas_path) == str:
            try:
                self.atlas_df = pd.read_csv(atlas_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find csv {atlas_path}. Make sure the path is correct and try again.")
        elif isinstance(atlas_path, (dict, pd.DataFrame)):
            self.atlas_df = pd.DataFrame(atlas_path) if isinstance(atlas_path, dict) else atlas_path
        else:
            raise ValueError("Invalid atlas_path type. Please provide a string path, a dictionary, or a DataFrame.")
        
        self.labels = self.atlas_df['region_name'].tolist()
    
    def name_at_index(self, index=[48, 94, 35]):
        """Returns the names of the regions at the given index.
        Args:
            index (list): List of three integers representing the x, y, z coordinates of the index (in voxel space).
        Returns:
            list: Names of the regions at the given index."""
        
        region_names = []
        for _, row in self.atlas_df.iterrows():
            mask_path = row['mask_path']
            mask_volume = nib.load(mask_path).get_fdata()
            if mask_volume[tuple(index)] > 0:
                region_names.append(row['region_name'])

        if len(region_names) == 0:
            return ["No matching region found"]
        else:
            return region_names

class CustomAtlasLabeler:
    """
    A class for labeling Nifti volumes based on predefined regions from an atlas.

    This class takes a Nifti volume and an atlas (CustomAtlas) and labels the volume
    based on the regions defined in the atlas. The labeling is done based on the 
    intensity thresholds specified for the Nifti volume.

    Attributes:
        atlas (CustomAtlas): An instance of the CustomAtlas class.
        labels (list): List of region names from the atlas.
        atlas_df (pd.DataFrame): DataFrame representing the atlas, with region names and mask paths.
        nifti (nib.Nifti1Image): The Nifti image to be labeled.
        volume_data (numpy.ndarray): Data from the Nifti image.
        min_threshold (int): Minimum intensity threshold for considering a voxel in the Nifti volume.
        max_threshold (int): Maximum intensity threshold for considering a voxel in the Nifti volume.
        labeled_data (pd.DataFrame or None): DataFrame containing labeled voxels after label_volume is called.
        voxel_counts (dict or None): Dictionary containing counts of labeled voxels per region after label_volume is called.
        unique_labels (list or None): List of unique labels assigned after label_volume is called.

    Args:
        nifti_path (str or nib.Nifti1Image): File path to the Nifti volume or a Nifti image object.
        atlas (CustomAtlas or str (path to atlas)): An instance of CustomAtlas or a file path to an atlas CSV file.
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Default is 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Default is 1.
    """

    def __init__(self, nifti_path, atlas=os.path.join(atlases_dir, "joseph_custom_atlas.csv"), min_threshold=1, max_threshold=1):
        
        if not isinstance(atlas, CustomAtlas):
            atlas = CustomAtlas(atlas)
        
        if not isinstance(nifti_path, nib.nifti1.Nifti1Image):
            try:
                nifti = nib.load(nifti_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find nifti {nifti_path}. Make sure the path is correct and try again.")

        self.atlas = atlas
        self.labels = self.atlas.labels
        self.atlas_df = atlas.atlas_df
        self.nifti = nifti
        self.volume_data = nifti.get_fdata()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.labeled_data = None
        self.voxel_counts = None
        self.unique_labels = None 
    
    def label_volume(self):
        """
        Labels the Nifti volume using the regions defined in the atlas.

        This method iterates through each voxel in the Nifti volume and labels it based
        on the overlapping regions in the atlas masks. Only voxels with intensity values
        within the specified threshold range are considered for labeling.

        The method updates the labeled_data, voxel_counts, and unique_labels attributes
        of the class with the results of the labeling process.

        Returns:
            CustomAtlasLabeler: The instance (itself) with updated labeled_data, voxel_counts, and unique_labels.
        """        
        
        # Initialize a dictionary to hold the labeling results
        results = {'index': [], 'atlas_label': []}
        
        # Identify voxels in the nifti volume that meet the threshold criteria
        within_threshold_voxels = set(zip(*np.where((self.volume_data >= self.min_threshold) & 
                                                    (self.volume_data <= self.max_threshold))))

        # Define a function to process each row of the DataFrame
        def process_row(row):
            region_name, mask_path = row['region_name'], row['mask_path']
            # Load the mask data
            mask_volume = nib.load(mask_path).get_fdata()
            # Find voxels in the mask that are non-zero
            mask_active_voxels = set(zip(*np.where(mask_volume > 0)))

            # Find intersection of within-threshold voxels and mask-active voxels
            intersecting_voxels = within_threshold_voxels.intersection(mask_active_voxels)
            for voxel_coords in intersecting_voxels:
                # Save the voxel coordinates and the corresponding label (region name) in the results
                results['index'].append(voxel_coords)
                results['atlas_label'].append(region_name)

        # Apply the function to each row of the DataFrame
        self.atlas_df.apply(process_row, axis=1)

        # Transform the collected results into a DataFrame
        results_df = pd.DataFrame(results)
        self.labeled_data = results_df
        self.voxel_counts = results_df['atlas_label'].value_counts().to_dict()
        self.unique_labels = results_df['atlas_label'].unique().tolist()
        return self

def label_csv_atlas(csv_path, roi_col_name="roi_2mm", min_threshold=1, max_threshold=1, atlas=os.path.join(atlases_dir,"HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")):
    """
    Labels a set of Nifti volumes specified in a CSV file or DataFrame using a provided atlas.

    This function processes either a CSV file or a DataFrame where each row corresponds to a Nifti volume.
    It labels each volume using the specified atlas and aggregates voxel counts for each region in the atlas.
    These counts are appended as new columns to the DataFrame.

    Args:
        csv_path (str or DataFrame): Path to the CSV file or a DataFrame containing information about the Nifti volumes.
                                     The CSV file or DataFrame is expected to have a column 'orig_roi_vol' containing
                                     paths to the Nifti volumes.
        atlas (Atlas or str): An instance of the Atlas class or a string path to the atlas file (.nii, .nii.gz, or .csv)
                              to be used for labeling the volumes.
        roi_col_name (str, optional): Name of the column in the CSV file containing the paths to the Nifti volumes.
                                        Defaults to "orig_roi_vol".
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
    
    Returns:
        pd.DataFrame: A DataFrame with the original data from the CSV file or DataFrame, augmented with voxel counts for
                      each region in the atlas for each Nifti volume. The voxel counts are added as new columns corresponding
                      to each region in the atlas.
    """
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    elif type(csv_path) == pd.DataFrame:
        df = csv_path
    else:
        raise ValueError(f"csv_path must be a string or a DataFrame. {type(csv_path)} was provided.")
    df['labeling_results'] = df[roi_col_name].progress_apply(lambda x: AtlasLabeler(x, atlas, min_threshold, max_threshold).label_volume())
    df['voxel_counts'] = df['labeling_results'].apply(lambda x: x.voxel_counts)
    labels = df['labeling_results'].iloc[0].labels
    for label in labels:
        df[label] = df['voxel_counts'].apply(lambda x: x.get(label, 0))
    df.drop(columns=['voxel_counts', 'labeling_results'], inplace=True)

    # Drop columns where the entire column's values are 0
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def label_csv_multi_atlas(csv_path, roi_col_name="roi_2mm", min_threshold=1, max_threshold=1, atlases=[os.path.join(atlases_dir,"atlases/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz"), os.path.join(atlases_dir,"atlases/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")]):
    """
    Labels a set of Nifti volumes specified in a CSV file or DataFrame using multiple provided atlases.

    This function processes either a CSV file or a DataFrame where each row corresponds to a Nifti volume.
    It applies labeling to each volume using multiple atlases specified in the atlases list and aggregates voxel 
    counts for each region in these atlases. The aggregated counts are then appended as new columns to the DataFrame.

    Args:
        csv_path (str or DataFrame): Path to the CSV file or a DataFrame containing information about the Nifti volumes.
                                     The CSV file or DataFrame is expected to have a column 'orig_roi_vol' containing
                                     paths to the Nifti volumes.
        atlases (list of Atlas or str): A list of Atlas instances or string paths to atlas files (.nii, .nii.gz, or .csv)
                                        to be used for labeling the volumes.
        roi_col_name (str, optional): Name of the column in the CSV file containing the paths to the Nifti volumes.
                                        Defaults to "orig_roi_vol".
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
    
    Returns:
        pd.DataFrame: A DataFrame with the original data from the CSV file or DataFrame, augmented with voxel counts for
                      each region in all the atlases for each Nifti volume. The voxel counts are added as new columns corresponding
                      to each region in the atlases.
    """
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    elif type(csv_path) == pd.DataFrame:
        df = csv_path
    else:
        raise ValueError(f"csv_path must be a string or a DataFrame. {type(csv_path)} was provided.")
    df['labeling_results'] = df[roi_col_name].progress_apply(lambda x: MultiAtlasLabeler(x, atlases, min_threshold, max_threshold).label_volume())
    df['voxel_counts'] = df['labeling_results'].apply(lambda x: x.voxel_counts)
    labels = df['labeling_results'].iloc[0].labels
    for label in labels:
        df[label] = df['voxel_counts'].apply(lambda x: x.get(label, 0))
    df.drop(columns=['voxel_counts', 'labeling_results'], inplace=True)
    
    # Drop columns where the entire column's values are 0
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def label_csv_custom_atlas(csv_path, roi_col_name="roi_2mm", min_threshold=1, max_threshold=1, atlas=os.path.join(atlases_dir,"joseph_custom_atlas.csv")):
    """
    Labels a set of Nifti volumes specified in a CSV file using a provided custom atlas.

    This function reads a CSV file, where each row corresponds to a Nifti volume, and applies
    labeling using the specified atlas. It extracts voxel counts for each region in the atlas and
    appends these counts as new columns in the DataFrame.

    Args:
        csv_path (str or DataFrame): Path to the CSV file containing information about the Nifti volumes.
                        The CSV file is expected to have a column 'roi_2mm' containing
                        paths to the Nifti volumes.
                        Alternatively, a DataFrame with the same structure as the CSV file can be provided.
        atlas (CustomAtlas or str): An instance of the CustomAtlas class to be used for labeling the volumes.
                        Alternatively, a string path to the csv file of the custom atlas can be provided.
                        This should be a CSV file with columns 'region_name' and 'mask_path'.
        roi_col_name (str, optional): Name of the column in the CSV file containing the paths to the Nifti volumes.
                        Defaults to "orig_roi_vol".
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
    
    Returns:
        pd.DataFrame: A DataFrame with the original data from the CSV file, augmented with 
                      voxel counts for each region in the atlas for each Nifti volume.
    """
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    elif type(csv_path) == pd.DataFrame:
        df = csv_path
    else:
        raise ValueError(f"csv_path must be a string or a DataFrame. {type(csv_path)} was provided.")
    df['labeling_results'] = df[roi_col_name].progress_apply(lambda x: CustomAtlasLabeler(x, atlas, min_threshold, max_threshold).label_volume())
    df['voxel_counts'] = df['labeling_results'].apply(lambda x: x.voxel_counts)
    labels = df['labeling_results'].iloc[0].labels
    for label in labels:
        df[label] = df['voxel_counts'].apply(lambda x: x.get(label, 0))
    df.drop(columns=['voxel_counts', 'labeling_results'], inplace=True)
    # Drop columns where the entire column's values are 0
    df = df.loc[:, (df != 0).any(axis=0)]
    return df