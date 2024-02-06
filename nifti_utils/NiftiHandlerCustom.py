from .NiftiHandler import NiftiHandler
import psycopg2
import gzip
import boto3
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import nibabel as nib
import numpy as np
import os
import pandas as pd
load_dotenv() 

class CustomStorage:

    session = boto3.session.Session()

    def save(self, name, content, bucket_name=None, max_length=None):
        # Retrieve bucket name from environment variables if not provided
        if bucket_name is None:
            bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')

        headers = {'ContentType': ''}
        s3 = self._get_s3_client()
        try:
            s3.upload_fileobj(content, bucket_name, name, ExtraArgs=headers)
        except Exception as e:
            print(f"Failed to upload: {e}")
            return None
        return name
    
    def get_file_from_cloud(self, cloud_filepath):
        bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')
        extension = cloud_filepath.split('.')[-1]
        
        client = self._get_s3_client()
        try:
            file_object = client.get_object(Bucket=os.getenv('AWS_STORAGE_BUCKET_NAME'), Key=cloud_filepath)
            file_data = file_object['Body'].read()
        except Exception as e:
            print(f"Failed to fetch: {e}")
            return None

        if extension == 'gz':
            fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=BytesIO(file_data)))
            return nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

        elif extension == 'npy':
            return np.load(BytesIO(file_data), allow_pickle=True)

        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _get_s3_client(self):
        return self.session.client('s3',
                                   region_name='nyc3',
                                   endpoint_url=os.getenv('AWS_S3_ENDPOINT_URL'),
                                   aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                   aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))


    def compress_nii_image(self, nii_img):
        """Compresses an NIfTI image.

        Args:
        - nii_img: An instance that has a to_bytes method which converts the image to byte data.

        Returns:
        - A BytesIO object containing the compressed image data.
        """
        img_data = nii_img.to_bytes()
        img_data_gz = BytesIO()
        with gzip.GzipFile(fileobj=img_data_gz, mode='w') as f_out:
            f_out.write(img_data)
        img_data_gz.seek(0)
        return img_data_gz

    def list_s3_files(self, s3_path):
        """
        List NIfTI files in an S3 bucket directory.

        Parameters
        ----------
        s3_path : str
            Path in the format s3://bucket-name/prefix
        
        Returns
        -------
        roi_paths : list of str
            List of S3 paths to NIfTI image ROIs.
        """

        # Parse the s3_path to extract bucket name and prefix
        if not s3_path.startswith('s3://'):
            raise ValueError("Provided path is not a valid S3 path.")
        
        s3_components = s3_path[5:].split('/', 1)
        bucket_name = s3_components[0]
        prefix = s3_components[1] if len(s3_components) > 1 else ""

        # Use the S3 client from the class's method
        s3 = self._get_s3_client()
        
        roi_paths = []
        
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if (obj['Key'].endswith('.nii') or obj['Key'].endswith('.nii.gz')):
                    roi_paths.append('s3://' + bucket_name + '/' + obj['Key'])
        
        return roi_paths 

class NiftiHandlerDjangoS3(NiftiHandler):
    """Extends NiftiHandler with postgres and S3 storage capabilities within a Django environment.
    Custom built for LesionBank project."""

    def __init__(self, data=None, model=None):
        """Initialize the NiftiHandler object with the provided data."""
        self.data = data
        self.model = model
    
    @classmethod
    def load(cls, data=None, model=None):
        """Create an instance of NiftiHandler and initialize it with the provided data."""
        # Create an instance of cls
        instance = cls()

        # Now use instance to set up everything except for handling the data.
        instance._setup_affines()  # Assuming this is still necessary
        instance._clear_properties()  # Reset properties
        instance.storage = CustomStorage()  # Setup storage
        if model is not None:
            instance.model = model
        instance.model = model if model is not None else None  # Setup model

        # After everything is set up, handle the input data.
        if data is None:
            if instance.data is None:
                print("No data provided for NiftiHandler initialization.")
                return
            data = instance.data
        if data is not None:
            instance._handle_input_data(data)
        
        return instance
    
    def _handle_input_data(self, data):
        """Direct input data to the appropriate processing method with enhancements for Django and S3."""
        if isinstance(data, str):
            self._handle_string_input(data)
        else:
            super()._handle_input_data(data)
        
    def _handle_string_input(self, path):
        """Attempt to load NIfTI data from various string-based sources, including local paths, S3, and databases."""
        try:
            if super()._try_populate_from_local(path): return
        except ValueError:
            pass  # If local fails, continue to try S3 and DB

        try:
            if self._try_populate_from_s3(path): return
            if self._try_populate_from_db(path): return
        except Exception as e:
            raise ValueError(f"Failed to load data from provided path or identifier: {path}. Error: {e}")

    def _try_populate_from_s3(self, path):
        """Attempt to load data from S3 storage."""
        try:
            nifti_obj = self._get_nifti_from_s3(path)
            self._populate_data_from_nifti(nifti_obj)
            return True
        except Exception:
            return False

    def _try_populate_from_db(self, identifier):
        """Attempt to load data from a database record, handling exceptions."""
        try:
            nifti_obj = self._get_nifti_from_db(identifier)
            self._populate_data_from_nifti(nifti_obj)
            return True
        except Exception as e:
            print(f"Failed to load from DB with identifier {identifier}: {e}")
            return False
        
    def _get_nifti_from_s3(self, s3_path):
        """Gets a NIfTI object from a file path in s3 storage, relative to the bucket root"""
        return self.storage.get_file_from_cloud(s3_path)

    def _get_nifti_from_db(self, id_name='lesion_id', id=None, model=None):
        if model is None:
            model = self.model
        if id is None or model is None:
            raise ValueError("ID and model must be provided.")

        query = f"""
        SELECT
            voxel_id, value
        FROM {model._meta.db_table}
        WHERE {id_name} = %s
        """
        params = (id,)
        result = self.run_raw_sql(query, params) # Fix this line to use Django ORM

        # Make a df_voxel_id dataframe (A dataframe with columns 'x', 'y', 'z', and 'value' (world space))
        self.df_voxel_id = pd.DataFrame(result, columns=['voxel_id', 'value'])

        # Make a df_xyz dataframe (A dataframe with columns 'Voxel ID' and 'value' (world space))
        voxel_id = self.df_voxel_id['voxel_id'].str.split('_', expand=True).astype(int)
        self.df_xyz = pd.concat([voxel_id, self.df_voxel_id['value']], axis=1)
        self.df_xyz.columns = ['x', 'y', 'z', 'value']

        # Make a 2D array with 4 columns: x, y, z, value (world space)
        self.twod_data = self.df_xyz.to_numpy()

        # Convert result to a 2D array with columns: x, y, z, value
        data = np.array([[*map(int, voxel_id.split('_')), value] for voxel_id, value in result])

        # Create a 3D array in voxel space and make it a NIfTI object
        return self.reshape_and_create_nifti(data)

    def save_to_s3(self, filename, file_content=None):
        """Saves a NIfTI object to s3 storage, relative to the bucket root"""
        if file_content is None:
            file_content = self.to_nifti_obj()
        file_content = self.storage.compress_nii_image(file_content)
        self.storage.save(filename, file_content)
        return filename
    
    def df_voxel_id_to_sql(self, id_name="upload_id", id_value=None, model=None, df=None):
        if model is None:
            model = self.model
        if df is None:
            if self.df_voxel_id is None:
                self.resample()
                self.reshape_to_2d()
                self.drop_zero_values()
                self.convert_to_dataframe()
            df = self.df_voxel_id

        if id_value is None:
            if self.id is None:
                id_value = str(int(datetime.now().timestamp()))
                self.id = id_value
            else:
                id_value = self.id

        df[id_name] = id_value

        data_to_insert = df.to_dict('records')

        # Insert to SQL
        batch_size = 1000
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            with transaction.atomic():
                # Create model objects within the atomic transaction and insert immediately
                model.objects.bulk_create(
                    [model(**{id_name: row[id_name], 'voxel_id': row['voxel_id'], 'value': row['value']}) for row in batch],
                    ignore_conflicts=True
                )

                print(f"{i} of {len(data_to_insert)} records inserted...")
            print(f"all {len(data_to_insert)} records successfully inserted.")

    def nifti_file_to_sql_wrapper(self, s3_path):
        """Wrapper function to convert a NIfTI file to SQL"""
        found_in_s3 = self._try_populate_from_s3(s3_path)
        if not found_in_s3:
            print(f"Failed to load data from provided path: {s3_path}")
            return None
        self.reshape_to_2d()
        self.drop_zero_values()
        self.convert_to_dataframe()
        self.df_voxel_id_to_sql()
    
    def sql_to_nifti_file_wrapper(self, id_name='lesion_id', id=None, model=None, filename=None):
        """Wrapper function to convert a SQL object to a NIfTI file"""
        if model is None or filename is None or id is None:
            raise ValueError("Model, filename, and id must be provided.")
        found_in_db = self._try_populate_from_db(id_name, id, model)
        if not found_in_db:
            print(f"Failed to load data from DB with identifier {id}.")
            return None
        self.save_to_s3(filename)
        return filename

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, '.env')
    load_dotenv(env_path)

    s3_path = "uploads/lesion_files/AliceWonderlandLesionNetwork_Hong_2010_Case01.nii.gz"
    nh = NiftiHandlerDjangoS3(s3_path)
    nh.normalize_to_quantile()
    nh.convert_to_dataframe()
    display(nh.df_xyz)

if __name__ == "__main__":
    main()