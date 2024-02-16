from scipy.stats import rankdata
from scipy.ndimage import zoom, measurements
from scipy import ndimage as ndi
from joblib import load
import nibabel as nib
import numpy as np
import os
import pandas as pd
import warnings
from .AnatomicalLabeler import Atlas, MultiAtlasLabeler

# Get the directory in which the current script is located
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
atlases_dir = os.path.join(parent_dir, 'atlas_data')

class NiftiHandler:
    """
    A versatile class designed to handle NIfTI images, capable of being initialized with various data types. 
    This class provides functionalities for loading, processing, and manipulating NIfTI data, 
    including conversion between different spatial representations and resolutions.

    Args:
        data (optional): Can be one of several types representing the NIfTI data:
            - A string path to a local file or a URI for S3/database storage (handled in derived classes).
            - A numpy array, either a 3D array representing voxel data or a 2D array with columns [x, y, z, value].
            - A pandas DataFrame with columns 'x', 'y', 'z', 'value' or 'voxel_id', 'value'.
            - A NiftiHandler object from which properties will be copied.
            - A nibabel NIfTI image object (nib.Nifti1Image or nib.Nifti2Image).

    Attributes:
        - nifti_obj (nib.Nifti1Image): The NIfTI image object representing the 3D volume.
        - data (numpy.ndarray): A 3D numpy array of the NIfTI image data.
        - affine (numpy.ndarray): The affine transformation matrix associated with the NIfTI image.
        - shape (tuple): The dimensions of the NIfTI image data.
        - resolution (str): The resolution of the NIfTI image, typically '1mm' or '2mm'.
        - twod_data (numpy.ndarray): A 2D array with columns for x, y, z coordinates and value.
        - df_xyz (pandas.DataFrame): DataFrame with columns 'x', 'y', 'z', and 'value'.
        - df_voxel_id (pandas.DataFrame): DataFrame with 'voxel_id' and 'value', where 'voxel_id' is 'x_y_z'.
        - id (str): An optional identifier for the NIfTI image.
        - type (str): The data type, either 'mask' or 'continuous'.
        - mask_applied (bool): Indicates if an anatomical mask has been applied.
        - is_quantile_normalized (bool): Indicates if the data has been normalized to quantile scores.
        - one_mm_affine, two_mm_affine (numpy.ndarray): Standard affine matrices for 1mm and 2mm resolutions.
        - data_obj: Same as data, for better interoperability with the nibabel library.

    User-facing methods:
        - load: Initialize the NiftiHandler object with the provided data.
        - reshape_and_create_nifti: Convert a 2D array with columns [x, y, z, value] into a 3D NIfTI image.
        - reshape_to_2d: Reshapes a 3d array to a 2d array with 4 columns: x, y, z, value.
        - drop_zero_values: Drops zero values from a 2d array with 4 columns: x, y, z, value.
        - convert_to_dataframe: Converts a 2D array with 4 columns: x, y, z, value into two pandas DataFrames.
        - resample: Resamples a 3D mask array to the specified resolution (1mm or 2mm).
        - apply_anatomical_mask: Applies an anatomical mask to a 3D array in voxel space.
        - normalize_to_quantile: Converts an n-dimensional array to quantile scores, excluding infs and nans.
        - to_nifti_obj: Converts a 3D array in voxel space to a NIfTI object (deprecated).
        - get_fdata: Returns the data attribute (for better interoperability with the nibabel library).
        - mask_and_flatten: Applies an anatomical mask to a 3D array in voxel space and flattens it.
    
    Internal methods:
        - _setup_affines: Define standard affine matrices for 1mm and 2mm resolutions.
        - _clear_properties: Reset properties to default values.
        - _handle_input_data: Process input data based on its type.
        - _handle_string_input: Attempt to load NIfTI data from various string-based sources.
        - _try_populate_from_local: Attempt to load data from a local file.
        - _handle_ndarray_input: Process numpy array inputs with different shapes and resolutions.
        - _handle_dataframe_input: Process DataFrame inputs based on their column structure.
        - _copy_properties: Copy properties from another NiftiHandler instance, including the nifti_obj.
        - _populate_data_from_nifti: Populate properties from a NIfTI image object.
        - _determine_data_type: Determine if the data represents a mask or continuous values.
        - _populate_data_from_voxel_id_dataframe: Populate properties from a DataFrame with voxel IDs and values.
        - _populate_data_from_2d_array: Populate the NiftiHandler object from a 2D array with x, y, z, value.
        - _determine_resolution_from_2d_array: Determine the resolution based on the range of x, y, z values in a 2D array.
    
    Example:
        To create a NiftiHandler object from a file path:
            nh = NiftiHandler('/path/to/nifti_file.nii.gz')
        To access the NIfTI data as a 3D numpy array:
            data_array = nh.data
    """
    
    def __init__(self, data=None):
        """Initialize the NiftiHandler object with the provided data."""
        self.data = data

    @classmethod
    def load(cls, data=None):
        """Create an instance of NiftiHandler and initialize it with the provided data."""
        # Create an instance of cls
        instance = cls(data)

        # If no data is provided to the load method, check if the instance has data
        if data is None:
            if instance.data is None:
                print("No data provided for NiftiHandler initialization.")
                return None

        # Since data is already assigned during instance creation, we can proceed
        instance._setup_affines()
        instance._clear_properties()
        instance._handle_input_data(data)
        
        return instance

    def _setup_affines(self):
        """Define standard affine matrices for 1mm and 2mm resolutions."""
        self.one_mm_affine = np.array([
            [-1., 0., 0., 90.],
            [0., 1., 0., -126.],
            [0., 0., 1., -72.],
            [0., 0., 0., 1.]
        ])
        self.two_mm_affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])

    def _clear_properties(self):
        """Reset properties to default values."""
        self.data = self.affine = self.shape = self.resolution = None
        self.twod_data = self.df_xyz = self.df_voxel_id = None
        self.id = self.type = None
        self.mask_applied = self.is_quantile_normalized = False

    def _handle_input_data(self, data):
        """Decide the appropriate method to handle the input data based on its type."""
        dispatch_map = {
            str: self._try_populate_from_local,
            np.ndarray: self._handle_ndarray_input,
            pd.DataFrame: self._handle_dataframe_input,
            NiftiHandler: self._copy_properties,
            nib.nifti1.Nifti1Image: self._populate_data_from_nifti,
            nib.nifti2.Nifti2Image: self._populate_data_from_nifti
        }
        handler = dispatch_map.get(type(data))
        if handler:
            handler(data)
        else:
            raise ValueError("Unsupported data type for NiftiHandler initialization.")
    
    def _handle_string_input(self, path):
        """Attempt to load NIfTI data from various string-based sources."""
        try:
            if self._try_populate_from_local(path): return
        except Exception as e:
            raise ValueError(f"Failed to load data from provided path or identifier: {path}. Error: {e}")

    def _try_populate_from_local(self, path):
        """Attempt to load data from a local file."""
        try:
            nifti_obj = nib.load(path)
            self._populate_data_from_nifti(nifti_obj)
            return True
        except Exception:
            return False

    def _handle_ndarray_input(self, array):
        """Process numpy array inputs with different shapes and resolutions."""
        if array.ndim == 2 and array.shape[1] == 4:
            self._populate_data_from_2d_array(array)
        elif array.shape in [(182, 218, 182), (91, 109, 91)]:
            resolution = '1mm' if array.shape == (182, 218, 182) else '2mm'
            nifti_obj = nib.Nifti1Image(array, getattr(self, f"{resolution}_affine"))
            self._populate_data_from_nifti(nifti_obj)
        else:
            raise ValueError("Unsupported ndarray shape for NiftiHandler initialization.")

    def _handle_dataframe_input(self, dataframe):
        """Process DataFrame inputs based on their column structure."""
        if set(dataframe.columns) == {'x', 'y', 'z', 'value'}:
            self._populate_data_from_2d_array(dataframe.to_numpy())
        elif set(dataframe.columns) == {'voxel_id', 'value'}:
            self._populate_data_from_voxel_id_dataframe(dataframe)
        else:
            raise ValueError("DataFrame must have columns 'x', 'y', 'z', 'value', or 'voxel_id' and 'value'.")

    def _copy_properties(self, other):
        """Copy properties from another NiftiHandler instance, including the nifti_obj."""
        attributes_to_copy = [
            'data', 'affine', 'shape', 'resolution', 'twod_data', 'df_xyz', 
            'df_voxel_id', 'id', 'type', 'mask_applied', 'is_quantile_normalized'
        ]
        for attr in attributes_to_copy:
            setattr(self, attr, getattr(other, attr))

    def flatten_along_last_axis(self):
        # Check if the last dimension of the shape is 1
        if self.shape[-1] == 1:
            # Remove the last dimension from data
            self.data = np.squeeze(self.data, axis=-1)
            # Update shape
            self.shape = self.data.shape

        return self

    def _populate_data_from_nifti(self, nifti_obj):
        """Populate properties from a NIfTI image object."""
        self.data = nifti_obj.get_fdata()
        self.affine = nifti_obj.affine
        self.shape = nifti_obj.shape
        
        # Flatten shape if the last dimension is 1
        if self.shape[-1] == 1:
            self.flatten_along_last_axis()
            
        # Determine resolution based on shape
        if self.shape == (182, 218, 182):
            self.resolution = '1mm'
        elif self.shape == (91, 109, 91):
            self.resolution = '2mm'
        else:
            self.resolution = "unknown"
            print(f"Unknown resolution. Expected (182, 218, 182) or (91, 109, 91), but got {self.shape}")

        # Determine the type of data (mask or continuous)
        self.type = self._determine_data_type(self.data)

    def _determine_data_type(self, data):
        """Determine if the data represents a mask or continuous values."""
        unique_values = np.unique(data)
        if set(unique_values).issubset({0, 1}):
            return 'mask'
        else:
            return 'continuous'

    def _populate_data_from_voxel_id_dataframe(self, df):
        """Populate properties from a DataFrame with voxel IDs and values."""
        # Convert 'voxel_id' to 'x', 'y', 'z'
        df_xyz = df['voxel_id'].str.split('_', expand=True).astype(int)
        df_xyz.columns = ['x', 'y', 'z']
        df_xyz['value'] = df['value']

        # Convert DataFrame to numpy array
        self.twod_data = df_xyz.to_numpy()

        # Populate the object from this 2D array
        self._populate_data_from_2d_array(self.twod_data)

    def _populate_data_from_2d_array(self, data):
        """Populate the NiftiHandler object from a 2D array with x, y, z, value."""
        # Determine the resolution based on the range of x, y, z values
        resolution = self._determine_resolution_from_2d_array(data)
        
        # Reshape 2D array to 3D based on resolution and populate data
        nifti_obj = self.reshape_and_create_nifti(data, resolution)
        self._populate_data_from_nifti(nifti_obj)

    def _determine_resolution_from_2d_array(self, data):
        """Determine the resolution based on the range of x, y, z values in a 2D array."""
        if np.any(data[:, :3] % 2 != 0):
            return '1mm'
        return '2mm'

        
    def reshape_and_create_nifti(self, nd_array, resolution='2mm'):
        """
        This method either converts a 2D array with columns [x, y, z, value] into a 3D NIfTI image or directly returns the NIfTI image if the input is already in the correct format. 
        It performs the necessary transformations from world space to voxel space using the specified resolution and applies the appropriate affine transformation.
        
        Parameters:
        - nd_array: A 2D numpy array with columns for x, y, z coordinates and value, or a Nifti1Image object.
        - resolution: A string indicating the resolution of the NIfTI image, either '1mm' or '2mm'.
        
        Returns:
        - A Nifti1Image object representing the 3D volume.
        
        Raises:
        - ValueError: If the resolution is not supported or if the nd_array shape is incorrect for direct Nifti1Image conversion.
        """
        
        # Check if nd_array is already a Nifti1Image
        if isinstance(nd_array, nib.nifti1.Nifti1Image):
            return nd_array
        
        # Validate nd_array is a 2D array with exactly 4 columns
        if not (isinstance(nd_array, np.ndarray) and nd_array.ndim == 2 and nd_array.shape[1] == 4):
            raise ValueError("Input nd_array must be a 2D array with exactly 4 columns.")
        
        # Setup affine matrix and determine the shape of the 3D array based on resolution
        if resolution == '2mm':
            affine_matrix = self.two_mm_affine
            three_d_array_shape = (91, 109, 91)
        elif resolution == '1mm':
            affine_matrix = self.one_mm_affine
            three_d_array_shape = (182, 218, 182)
        else:
            raise ValueError("Unsupported resolution. Expected '1mm' or '2mm'.")
        
        # Initialize an empty 3D array
        three_d_array = np.zeros(three_d_array_shape, dtype=nd_array.dtype)
        
        # Convert world space coordinates to voxel space
        worldspace_coords = np.hstack((nd_array[:, :3], np.ones((nd_array.shape[0], 1))))
        inverse_matrix = np.linalg.inv(affine_matrix)
        voxel_coords = np.dot(worldspace_coords, inverse_matrix.T)[:, :3]
        voxel_coords = np.round(np.clip(voxel_coords, a_min=0, a_max=np.array(three_d_array_shape) - 1)).astype(int)
        
        # Populate the 3D array with values
        three_d_array[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = nd_array[:, 3]
        
        # Cast three_d_array as int16 to save space
        three_d_array = three_d_array.astype(np.int16)

        # Create and return a Nifti1Image object
        return nib.Nifti1Image(three_d_array, affine_matrix)
    
    def reshape_to_2d(self, ndarray=None):
        """
            Transforms a 3D brain volume (ndarray) into a 2D array of coordinates and values, 
            presenting the data in world space. The resulting 2D array consists of four columns: x, y, z, and value. 
            This process involves filtering out zero-value points from the 3D volume to focus on significant data points, 
            converting voxel space coordinates to world space coordinates using the affine transformation defined for the volume, 
            and rounding the intensity values for precision control.
            
            The method uses a predefined affine transformation (self.two_mm_affine) to convert voxel coordinates 
            into world space coordinates, assuming a 2mm resolution by default. Adjustments may be required 
            for volumes with different resolutions or affine transformations.

            Args:
                ndarray (numpy.ndarray, optional): A 3D numpy array representing a brain volume in voxel space. 
                    If not provided, the method will use the class's own data attribute.

            Returns:
                self: Returns the instance itself, with the `twod_data` attribute updated to contain the 
                    generated 2D array.

            Example:
                Given a NiftiHandler instance `nh` with a loaded 3D brain volume, you can generate and access 
                the 2D representation as follows:
                    nh.reshape_to_2d()
                    twod_data = nh.twod_data  # Access the 2D array with x, y, z, and value columns
            """        
        if ndarray is None:
            ndarray = self.data # 3d array in voxel space
        non_zero_indices = np.nonzero(ndarray)
        values = np.round(ndarray[non_zero_indices], 3)  # Round the values to 3 decimals
        coords = np.array(non_zero_indices).T
        forward_matrix = self.two_mm_affine[:3, :3]
        forward_translation = self.two_mm_affine[:3, 3]
        transformed_coords = np.dot(coords, forward_matrix.T) + forward_translation
        self.twod_data = np.column_stack((transformed_coords, values)) # Swap the order of the columns
        return self
    
    def drop_zero_values(self, ndarray=None):
        """Drops zero values from a 2d array with 4 columns: x, y, z, value"""
        if ndarray is None:
            ndarray = self.twod_data
        mask = ~np.logical_or(ndarray[:, 3] == 0, np.isnan(ndarray[:, 3]))  # create a mask that is False where the first column is 0 or NaN
        self.twod_data = ndarray[mask]
        return self
    
    def convert_to_dataframe(self, nd_array=None):
        """
        Converts a 2D numpy array into two distinct pandas DataFrames for different representations of the same data.
        This conversion facilitates data analysis and manipulation within the pandas ecosystem, 
        especially for neuroimaging data where spatial coordinates and intensity values are key.

        The first DataFrame (`df_xyz`) includes columns for spatial coordinates ('x', 'y', 'z') and intensity values ('value'), 
        representing each non-zero voxel in the original 3D volume. This format is direct and intuitive for spatial analyses.

        The second DataFrame (`df_voxel_id`) combines the spatial coordinates into a single 'Voxel ID' string column, 
        with the format 'x_y_z', alongside the corresponding 'value' column. This format can be particularly useful for 
        indexing, merging with other data sets, or when a compact representation of voxel locations is required.

        Args:
            nd_array (numpy.ndarray, optional): A 2D numpy array with four columns representing 'x', 'y', 'z' spatial coordinates, 
                and 'value' for each voxel's intensity. If not provided, the method attempts to use the `twod_data` attribute 
                of the instance, generating it if necessary.

        Raises:
            ValueError: If the input array does not have exactly four columns.

        Returns:
            self: The instance itself, with the `df_xyz` and `df_voxel_id` attributes updated to contain the generated DataFrames.

        Example:
            Assuming `nh` is an instance of NiftiHandler with 3D volume data already processed:
                nh.convert_to_dataframe()
                df_xyz, df_voxel_id = nh.df_xyz, nh.df_voxel_id
            Here, `df_xyz` can be used for spatial analysis, while `df_voxel_id` is useful for compact data representation or merging.

        Note:
            This method assumes the input nd_array represents non-zero voxels in a 3D volume. It's crucial for the array to be pre-processed 
            (e.g., via `reshape_to_2d`) to ensure it contains only relevant data points, as zero-value voxels are typically omitted 
            from the 2D representation.
        """
        if nd_array is None:
            if self.twod_data is None:
                self.reshape_to_2d()
                self.drop_zero_values()
            nd_array = self.twod_data

        if nd_array.shape[1] != 4:
            raise ValueError("Array must have exactly 4 columns")

        # Include 'value' column in df_xyz
        df_xyz = pd.DataFrame(nd_array, columns=['x', 'y', 'z', 'value'])

        # Create voxel_id and include 'value' column in df_voxel_id
        voxel_id = ['{}_{}_{}'.format(*map(int, row[:3])) for row in nd_array]
        df_voxel_id = pd.DataFrame({'voxel_id': voxel_id, 'value': nd_array[:, 3]})

        self.df_xyz = df_xyz
        self.df_voxel_id = df_voxel_id
        return self
    
    def resample(self, target_resolution="2mm", nd_array=None):
        """Resamples a 3D mask array to the specified resolution.
        The nd_array is a 3D array in voxel space. This function supports resampling
        from 1mm to 2mm and from 2mm to 1mm resolutions. It ensures that all non-zero
        values in the resampled array are set to 1. 

        Args:
            target_resolution (str): The target resolution, either '1mm' or '2mm'.
            nd_array (numpy.ndarray, optional): The array to resample. If None, uses the instance's data.

        Returns:
            numpy.ndarray: The resampled array.
        Note:
            Only works well for binary masks, not for continuous data.
        """
        if target_resolution not in ['1mm', '2mm']:
            raise ValueError("Target resolution must be '1mm' or '2mm'.")

        if nd_array is None:
            nd_array = self.data

        if self.resolution == target_resolution:
            print(f"This array is already in {target_resolution} resolution.")
            return self

        if self.type != 'mask':
            print("This method is intended for binary masks. It may not work as expected for continuous data.")

        valid_transitions = [('1mm', '2mm'), ('2mm', '1mm')]
        if (self.resolution, target_resolution) not in valid_transitions:
            raise ValueError(f"Cannot resample from {self.resolution} to {target_resolution}.")

        # Determine resampling factor based on the desired transition
        if target_resolution == '2mm':
            resample_factor = np.diag(self.one_mm_affine)[:3] / np.diag(self.two_mm_affine)[:3]
        else:  # Resampling from 2mm to 1mm
            resample_factor = np.diag(self.two_mm_affine)[:3] / np.diag(self.one_mm_affine)[:3]

        # Ensure resample_factor is an array of floats
        resample_factor = np.array(resample_factor, dtype=float)

        # Resample the array
        resampled_nd_array = zoom(nd_array, resample_factor, order=1)  # order=1 for linear interpolation

        # Post-process to ensure binary values (0 or 1)
        resampled_nd_array[resampled_nd_array != 0] = 1

        # Update instance attributes
        self.data = resampled_nd_array
        self.resolution = target_resolution
        self.shape = resampled_nd_array.shape

        return self
    
    def apply_anatomical_mask(self, mask_filepath=f"{current_dir}/MNI152_T1_2mm_brain_mask.nii.gz", nd_array=None):
        """
        Applies an anatomical mask to the specified 3D numpy array representing brain volume data in voxel space. 
        The mask is used to filter or isolate regions of interest within the brain volume, setting voxel values 
        outside the mask to NaN and ensuring voxel values inside the mask are valid numbers (replacing NaNs and 
        infinities with 0).

        This method is particularly useful for focusing analyses on brain regions covered by the mask, 
        which typically corresponds to brain tissue, excluding non-brain areas.

        Args:
            mask_filepath (str, optional): The file path to the anatomical mask in NIfTI format. Defaults to 
                "MNI152_T1_2mm_brain_mask.nii.gz", which is a commonly used brain mask in the MNI152 standard space.
            nd_array (numpy.ndarray, optional): A 3D numpy array representing the brain volume data to be masked. 
                If not provided, the method will use the instance's `data` attribute.

        Raises:
            ValueError: If the ndarray and the mask do not have the same shape, indicating a potential misalignment 
                that could lead to incorrect masking.

        Returns:
            self: The instance itself, with the `data` attribute updated to the masked brain volume.

        Example:
            Assuming `nh` is an instance of NiftiHandler with loaded 3D brain volume data:
                nh.apply_anatomical_mask()
            This modifies `nh.data` to only include voxel values within the brain region defined by the default mask.

        Note:
            It is crucial that the ndarray and the mask have the same dimensions and are aligned in the same anatomical space.
            If there's a shape mismatch, consider resampling the mask or the ndarray to ensure they align correctly before masking.
        """        
        if nd_array is None:
            nd_array = self.data
        
        mask = nib.load(mask_filepath).get_fdata()
        if nd_array.shape != mask.shape:
            print(f"Array and mask must have the same shape. Got {nd_array.shape} and {mask.shape}.")
            self.resample()
            nd_array = self.data
            if nd_array.shape != mask.shape:
                raise ValueError(f"Array and mask must have the same shape. Got {nd_array.shape} and {mask.shape}.")
        nd_array[mask == 0] = np.nan
        nd_array[mask == 1] = np.nan_to_num(nd_array[mask == 1], nan=0, posinf=0, neginf=0)
        self.mask_applied = True
        self.data = nd_array
        return self
    
    def mask_and_flatten(self, nd_array=None, mask_filepath=f"{current_dir}/MNI152_T1_2mm_brain_mask.nii.gz"):
        """Applies an anatomical mask to a 3D array in voxel space"""
        if nd_array is None:
            nd_array = self.data
            if nd_array is None:
                raise ValueError("Array must be provided.")
        mask = nib.load(mask_filepath).get_fdata()
        if nd_array.shape != mask.shape:
            raise ValueError("Array and mask must have the same shape.")
        # Apply the mask and flatten the array
        nd_array = nd_array[mask == 1].flatten()
        return nd_array

    def normalize_to_quantile(self, nd_array=None):
        """
        Transforms the values of an n-dimensional numpy array into their corresponding quantile scores, 
        effectively normalizing the distribution of values. This method is particularly useful for 
        standardizing brain imaging data, making it more comparable across subjects or scans by 
        converting voxel intensity values into a uniform scale of quantiles. Infinities and NaN values 
        are excluded from the calculation to ensure only meaningful, finite data contributes to the normalization.

        Prior to normalization, if an anatomical mask has not been applied to the data, this method will 
        automatically apply one to focus normalization on brain tissue voxels. This ensures that the 
        transformation is meaningful and applied consistently.

        Args:
            nd_array (numpy.ndarray, optional): An n-dimensional numpy array representing the brain volume data 
                to be normalized. If not provided, the method will use the instance's `data` attribute, assuming 
                it has been previously loaded and optionally masked.

        Raises:
            ValueError: If there's a shape mismatch between the input and output data, indicating an error in 
                processing or data handling.

        Returns:
            self: The instance itself, with the `data` attribute updated to the normalized brain volume.

        Example:
            Assuming `nh` is an instance of NiftiHandler with loaded 3D brain volume data:
                nh.normalize_to_quantile()
            After calling this method, `nh.data` will contain the quantile-normalized values of the original volume data.

        Note:
            This method assumes the input data is finite and meaningful brain volume data. Infinities and NaNs are 
            ignored in the normalization process, and it is recommended to apply an anatomical mask before normalization 
            to ensure only relevant brain voxels are included.
        """
        if nd_array is None:
            if not self.mask_applied:
                self.apply_anatomical_mask()
            nd_array = self.data
        
        original_shape = nd_array.shape
        
        # Mask finite values
        finite_mask = np.isfinite(nd_array)

        # Flatten finite values and calculate quantile scores
        data_flat = nd_array[finite_mask].flatten()
        data_ranked = rankdata(data_flat)
        data_quantile_scores = data_ranked / len(data_ranked)

        # Initialize output array with NaNs and populate with quantile scores
        output_array = np.full_like(nd_array, np.nan, dtype=np.float64)
        output_array[finite_mask] = data_quantile_scores

        # Ensure output shape consistency
        final_shape = output_array.shape
        if final_shape != original_shape:
            raise ValueError(f"Shape mismatch. Expected {original_shape}, got {final_shape}.")

        self.is_quantile_normalized = True
        self.data = output_array

        return self

    def to_nifti_obj(self, data=None, resolution='2mm'):
        """Deprecated method, use nifti_obj property instead.
        Converts a 3D array in voxel space to a NIfTI object"""
        if data is None:
            data = self.data
        if resolution == '2mm':
            return nib.Nifti1Image(data, self.two_mm_affine)
        elif resolution == '1mm':
            return nib.Nifti1Image(data, self.one_mm_affine)

    def load_components(self, model_path=f'{current_dir}/model.joblib', pca_path=f'{current_dir}/pca.joblib'):
        model = load(model_path)
        pca = load(pca_path)
        return model, pca

    def predict(self):
        self.normalize_to_quantile()
        data = self.mask_and_flatten()
        model, pca = self.load_components()
        
        new_data_pca = pca.transform([data])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            prediction = model.predict(new_data_pca)
        return prediction

    def predict_with_confidence(self):
        self.normalize_to_quantile()
        data = self.mask_and_flatten()
        model, pca = self.load_components()
        
        new_data_pca = pca.transform([data])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            # Get the predicted probabilities for each class
            probabilities = model.predict_proba(new_data_pca)[0]  # [0] to get the probabilities for the sample

        # Pair each class label with its corresponding probability
        class_probabilities = list(zip(model.classes_, probabilities))

        # Sort the classes based on the probability in descending order
        ranked_predictions = sorted(class_probabilities, key=lambda x: x[1], reverse=True)
        
        return ranked_predictions

    def world_to_voxel_space(self, world_coordinates, affine=None):
        if affine is None:
            affine = self.affine
        """
        Converts world space coordinates to voxel space using the inverse of the affine matrix.
        
        Parameters:
        - world_coordinates: The coordinates in world space to be converted (3-element sequence or Nx3 array).
        - affine: The affine transformation matrix used to map voxel space to world space (4x4 array).
        
        Returns:
        - voxel_coordinates: The equivalent coordinates in voxel space.
        """
        # Calculate the inverse of the affine matrix
        inverse_affine = np.linalg.inv(affine)
        # Apply the inverse affine transformation to convert world coordinates to voxel space
        voxel_coordinates = nib.affines.apply_affine(inverse_affine, world_coordinates)
        return tuple(np.rint(voxel_coordinates).astype(int))

    def voxel_to_world_space(self, voxel_coordinates, affine=None):
        """
        Converts voxel space coordinates to world space using the affine matrix.
        
        Parameters:
        - voxel_coordinates: The coordinates in voxel space to be converted (3-element sequence or Nx3 array).
        - affine: The affine transformation matrix used to map voxel space to world space (4x4 array).
        
        Returns:
        - world_coordinates: The equivalent coordinates in world space.
        """
        if affine is None:
            affine = self.affine
        # Apply the affine transformation to convert voxel coordinates to world space
        world_coordinates = nib.affines.apply_affine(affine, voxel_coordinates)
        return tuple(np.rint(world_coordinates).astype(int))
    
    def local_maxima_3D(self, data=None, order=10):
        """Detects local maxima in a 3D array

        Parameters
        ---------
        data : 3d ndarray
        order : int
            How many points on each side to use for the comparison

        Returns
        -------
        coords : ndarray
            coordinates of the local maxima
        values : ndarray
            values of the local maxima
        """
        if data is None:
            data = self.data
        size = 1 + 2 * order
        footprint = np.ones((size, size, size))
        footprint[order, order, order] = 0

        filtered = ndi.maximum_filter(data, footprint=footprint)
        mask_local_maxima = data > filtered
        coords = np.asarray(np.where(mask_local_maxima)).T
        values = data[mask_local_maxima]

        df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
        df['value'] = values
        df.sort_values(by='value', inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)
        return df

    def local_minima_3D(self, data=None, order=10):
        """Detects local minima in a 3D array

        Parameters
        ---------
        data : 3d ndarray
        order : int
            How many points on each side to use for the comparison

        Returns
        -------
        coords : ndarray
            coordinates of the local minima
        values : ndarray
            values of the local minima
        """
        if data is None:
            data = self.data
        size = 1 + 2 * order
        footprint = np.ones((size, size, size))
        footprint[order, order, order] = 0

        filtered = ndi.minimum_filter(data, footprint=footprint)
        mask_local_minima = data < filtered
        coords = np.asarray(np.where(mask_local_minima)).T
        values = data[mask_local_minima]

        df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
        df['value'] = values
        df.sort_values(by='value', inplace=True, ascending=False) # How can I also reset the index to reflect the new order?

        df.reset_index(drop=True, inplace=True)

        return df

    def get_local_extrema(self, order=10, data=None):
        if data is None:
            data = self.data
        df_max = self.local_maxima_3D(data=data, order=order)
        df_min = self.local_minima_3D(data=data, order=order)
        df = pd.concat([df_max, df_min], axis=0, ignore_index=True)
        return df

    @property
    def center_of_mass(self, data=None):
        if data is None:
            data = self.data
        data = np.nan_to_num(self.data)  # Replace NaNs with 0s
        return measurements.center_of_mass(data)
    
    @property
    def center_of_mass_world_space(self, data=None):
        if data is None:
            data = self.data
        return self.voxel_to_world_space(self.center_of_mass, affine=self.affine)

    @property
    def peak_coordinate_voxel_space(self):
        """Returns the peak coordinate in voxel space"""
        # return np.unravel_index(np.argmax(self.data, axis=None), self.data.shape)
        return np.unravel_index(np.nanargmax(self.data, axis=None), self.data.shape)
    
    @property
    def peak_coordinate_world_space(self):
        """Returns the peak coordinate in world space"""
        peak_voxel = self.peak_coordinate_voxel_space
        return self.voxel_to_world_space(peak_voxel)
    
    @property
    def peak_value(self):
        """Returns the peak value"""
        return np.nanmax(self.data)

    @property
    def anatomical_name_at_peak(self):
        """Returns the anatomical name at the peak coordinate"""
        index = self.peak_coordinate_voxel_space
        atlas_cort_name = [Atlas(os.path.join(atlases_dir, "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")).name_at_index(index)]
        atlas_sub_name = [Atlas(os.path.join(atlases_dir, "HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")).name_at_index(index)]
        results = [x for x in list(set(atlas_cort_name + atlas_sub_name)) if x != 'No matching region found']
        if len(results) == 0:
            return "No matching region found"
        elif len(results) == 1:
            return results[0]
        return results
    
    @property
    def anatomical_name_at_center_of_mass(self):
        """Returns the anatomical name at the center of mass"""
        index = self.center_of_mass
        index = tuple(map(int, index))
        atlas_cort_name = [Atlas(os.path.join(atlases_dir, "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")).name_at_index(index)]
        atlas_sub_name = [Atlas(os.path.join(atlases_dir, "HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")).name_at_index(index)]
        results = [x for x in list(set(atlas_cort_name + atlas_sub_name)) if x != 'No matching region found']
        if len(results) == 0:
            return "No matching region found"
        elif len(results) == 1:
            return results[0]
        return results
    
    @property
    def nifti_obj(self):
        if self.resolution == '2mm':
            self.nifti_obj = nib.Nifti1Image(self.data, self.two_mm_affine)
            return nib.Nifti1Image(self.data, self.two_mm_affine)
        elif self.resolution == '1mm':
            self.nifti_obj = nib.Nifti1Image(self.data, self.one_mm_affine)
            return nib.Nifti1Image(self.data, self.one_mm_affine)
    
    """Classes for better interoperability with the nibabel library"""
    def get_fdata(self):
        return self.data
    @property
    def data_obj(self):
        return self.data