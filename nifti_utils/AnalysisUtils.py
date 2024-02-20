import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib
from tqdm import tqdm
tqdm.pandas()

from .NiftiHandler import NiftiHandler as nh

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from joblib import dump


current_dir = os.path.dirname(__file__)
    
class ImageComparison:
    """A class for comparing two NIfTI images. Can compute Pearson correlation and Dice coefficient."""
    def __init__(self):
        pass

    def correlate_images(self, image1, image2):
        """Computes the Pearson correlation between two NIfTI images."""
        # Check if both images are instances of NiftiHandler
        if not isinstance(image1, nh):
            image1 = nh.load(image1)     
        if not isinstance(image2, nh):
            image2 = nh.load(image2)
        # Check if resolutions match, if not, resample
        if image1.resolution != image2.resolution:
            image1.resample()
            image2.resample()
        # Ensure shapes are the same
        if image1.data.shape != image2.data.shape:
            raise ValueError("The images must have the same shape.")
        # Check if types match (mask or continuous)
        if image1.type != image2.type:
            raise ValueError("Both images must be of the same type (mask or continuous).")
        # Ensure both images are quantile normalized
        if not image1.is_quantile_normalized:
            image1.normalize_to_quantile()
        if not image2.is_quantile_normalized:
            image2.normalize_to_quantile()
        # Flatten the 3D arrays to 1D to compute correlation
        flattened_data1 = image1.data.flatten()
        flattened_data2 = image2.data.flatten()

        # Ensure no infs or NaNs
        flattened_data1 = flattened_data1[np.isfinite(flattened_data1)]
        flattened_data2 = flattened_data2[np.isfinite(flattened_data2)]

        # Check if after removing infs and NaNs, the arrays are non-empty and of the same length
        if flattened_data1.size == 0 or flattened_data2.size == 0:
            raise ValueError("One or both of the images result in empty arrays after removing infs and NaNs.")
        if flattened_data1.size != flattened_data2.size:
            raise ValueError("The images must have the same number of finite values.")

        # Compute Pearson correlation
        correlation, _ = stats.pearsonr(flattened_data1, flattened_data2)
        
        return correlation

    def correlate_image_with_list_as_df(self, image1, image_list):
        """Computes the Pearson correlation between one image and a list of images, returning a DataFrame."""
        # Initialize a list to hold each row's data
        results_list = []
        # Wrap list_of_images with tqdm for a progress bar
        list_of_images = image_list.copy()
        for image2 in tqdm(list_of_images, desc="Correlating images"):
            try:
                # Compute the correlation
                correlation = self.correlate_images(image1, image2)
                # Append the result as a dictionary to the list
                results_list.append({"image2": image2, "correlation": correlation})
            except ValueError as e:
                # Handle errors by appending an error indication or NaN for correlation
                print(f"Error correlating {image1} with {image2}: {e}")
                results_list.append({"image2": image2, "correlation": pd.NA})

        # Convert the list of dictionaries to a DataFrame
        results_df = pd.DataFrame(results_list)
        return results_df

    def dice_coefficient(self, image1, image2):
        """
        Computes the Dice coefficient between two NIfTI images.

        Parameters:
        image1 (NiftiHandler): The first NIfTI image.
        image2 (NiftiHandler): The second NIfTI image.

        Returns:
        float: The Dice coefficient between the two images.
        """
        try:
            # Check if both images are instances of NiftiHandler
            if not isinstance(image1, nh) or not isinstance(image2, nh):
                raise TypeError("Both images must be instances of NiftiHandler.")
            # Check if resolutions match, if not, resample
            if image1.resolution != image2.resolution:
                image1.resample()
                image2.resample()
            # Ensure shapes are the same
            if image1.data.shape != image2.data.shape:
                raise ValueError("The images must have the same shape.")
            # Check if types match (mask or continuous)
            if image1.type != 'mask' or image1.type != image2.type:
                raise ValueError("Both images must be of the type mask")
            
            intersection = np.logical_and(image1.data, image2.data)
            dice_coefficient = 2 * np.sum(intersection) / (np.sum(image1.data) + np.sum(image2.data))
            return dice_coefficient
        
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def correlate_with_symptoms(self, image, data_to_compare):
        """
        Correlates a given image with a set of data and performs statistical analysis 
        to find significant differences in correlations across symptoms.

        Parameters:
        image (str or NiftiImage): The image or path to the image to be correlated.
        data_to_compare (list or pd.DataFrame): Data containing symptoms and paths 
                                                to images for comparison.

        Returns:
        pd.DataFrame: A summary DataFrame with symptoms ranked by mean correlation 
                    and statistical analysis results.
        """
        # DataFrame creation from input data
        df = pd.DataFrame(data_to_compare) if isinstance(data_to_compare, list) else data_to_compare

        # Validate DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Data must be a list of dictionaries or a pandas DataFrame.")

        # If the original image isn't already a NiftiHandler, convert it; otherwise use it as is
        if not isinstance(image, nh):
            original_image = nh.load(image)
        else:
            original_image = image
        original_image.normalize_to_quantile()

        # Calculate correlations and z-values
        df['correlation'] = df['path'].apply(lambda x: self.correlate_images(original_image, nh(x)))
        df['z_value'] = np.arctanh(df['correlation'])

        # Perform ANOVA and Tukey's HSD test if significant
        description, summary_df = self.anova_and_tukey(df)

        return description, summary_df

    def anova_and_tukey(self, df, label="symptom"):
        """
        Performs statistical analysis (ANOVA and Tukey's HSD test) on the given DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame containing symptom, correlation, and z_value data.

        Returns:
        Tuple: A description string and a pd.DataFrame with statistical analysis results.
        """
        anova_results = ols(f'z_value ~ C({label})', data=df).fit()
        anova_table = sm.stats.anova_lm(anova_results, typ=2)
        p_value = anova_table['PR(>F)'].iloc[0]
        if p_value >= 0.05:
            
            description = f"ANOVA (p={p_value:.3f}) found no significant differences among symptom correlations."

            # Create a DataFrame with the same structure as from Tukey's HSD test
            mean_corr = df.groupby(label)['correlation'].mean().sort_values(ascending=False)
            summary_df = pd.DataFrame({
                label: mean_corr.index,
                "mean_correlation(r)": mean_corr.values,
                f"{label}s_with_significant_differences": [[] for _ in range(len(mean_corr))],
                "percentage_of_comparisons_that_are_significant": [None] * len(mean_corr)
            })

            return description, summary_df

        tukey_results = pairwise_tukeyhsd(df['z_value'], df[label])
        description = f"""ANOVA (p={p_value}) followed by Tukey's HSD test found significant differences among {label} correlations."""

        return description, self.process_tukey_results(df, tukey_results)

    def process_tukey_results(self, df, tukey_results):
        """
        Processes Tukey's HSD test results and creates a summary DataFrame.

        Parameters:
        df (pd.DataFrame): Original DataFrame used for Tukey's test.
        tukey_results (TukeyHSDResults): Results from Tukey's HSD test.

        Returns:
        pd.DataFrame: Processed summary DataFrame with detailed statistical analysis.
        """
        mean_corr = df.groupby('symptom')['correlation'].mean()
        tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

        rows = []
        for symptom in df['symptom'].unique():
            symptom_data = self.calculate_symptom_statistics(symptom, tukey_df, mean_corr)
            rows.append(symptom_data)

        return pd.DataFrame(rows).sort_values(by="mean_correlation(r)", ascending=False)

    def calculate_symptom_statistics(self, symptom, tukey_df, mean_corr):
        """
        Calculates statistical measures for a given symptom based on Tukey's results.

        Parameters:
        symptom (str): The symptom to calculate statistics for.
        tukey_df (pd.DataFrame): DataFrame with Tukey's test results.
        mean_corr (pd.Series): Series with mean correlations for each symptom.

        Returns:
        dict: A dictionary with calculated statistical measures for the symptom.
        """
        def get_other_symptom(row, symptom):
            return row['group1'] if row['group2'] == symptom else row['group2']
        
        symptom_pairs = tukey_df[(tukey_df['group1'] == symptom) | (tukey_df['group2'] == symptom)]

        significant_symptoms = symptom_pairs[symptom_pairs['reject'] == True].copy()  # Use .copy() here

        # Applying the function to the DataFrame
        significant_symptoms['other_symptom'] = significant_symptoms.apply(get_other_symptom, axis=1, args=(symptom,))

        # Creating the dictionary
        significant_symptoms_dict = significant_symptoms.set_index('other_symptom')['p-adj'].to_dict()
        
        prop_significant = len(significant_symptoms) / len(symptom_pairs) * 100

        return {
            "symptom": symptom,
            "mean_correlation(r)": mean_corr[symptom],
            "symptoms_with_significant_differences": significant_symptoms_dict,
            "percentage_of_comparisons_that_are_significant": prop_significant
        }

class GroupAnalysis:

    def __init__(self, data, label="symptom"):
        self.label = label
        if type(data) != pd.DataFrame:
            if type(data) != list:
                raise TypeError("Data must be a list of dictionaries.")
            self.df = pd.DataFrame(data)
        else: 
            self.df = data
        # For testing, let's randomly select five rows from the dataframe
        # self.df = self.df.sample(20)
        # # Reset index
        # self.df.reset_index(drop=True, inplace=True)
    def get_nifti_from_s3(self, s3_path):
        """Gets a NIfTI object from a file path in s3 storage, relative to the bucket root"""
        return self.storage.get_file_from_cloud(s3_path)
    
    def save_to_s3(self, filename, resolution='2mm', file_content=None):
        """Saves a NIfTI object to s3 storage, relative to the bucket root"""
        if file_content is None:
            file_content = self.data
            file_content = self.to_nifti_obj(file_content, resolution)
        file_content = self.storage.compress_nii_image(file_content)
        self.storage.save(filename, file_content)
        return filename
    
    def mask_and_flatten(self, nd_array, mask_filepath=f"{current_dir}/MNI152_T1_2mm_brain_mask.nii.gz"):
        """Applies an anatomical mask to a 3D array in voxel space"""
        if nd_array is None:
            raise ValueError("Array must be provided.")
        
        mask = nib.load(mask_filepath).get_fdata()
        if nd_array.shape != mask.shape:
            raise ValueError(f"Array and mask must have the same shape. Got {nd_array.shape} and {mask.shape} respectively.")
        # Apply the mask and flatten the array
        nd_array = nd_array[mask == 1].flatten()
        return nd_array

    def sensitivity_analysis(self):
        pass

    def specificity_analysis(self):
        pass
    
    def mlogit_loocv(self):
        df = self.df.copy()
        pca_df, explained_variance = self.pca(n_components=8, df=df)
        print(f"Explained variance: {explained_variance}")

        # Initialize LeaveOneOut
        loo = LeaveOneOut()

        # Prepare dataset
        X = pca_df.drop(self.label, axis=1)
        y = pca_df[self.label]

        # List to store results
        results = []

        # Perform Leave-One-Out Cross Validation
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Create and train the logistic regression model
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Add results to the list
            results.append({
                'Accuracy': accuracy,
                'Confusion Matrix': conf_matrix,
                'Classification Report': class_report,
                f'Actual {self.label}': y_test.iloc[0],
                f'Predicted {self.label}': y_pred[0]
            })

        # Create DataFrame from results
        results_df = pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)
        accuracy_summary = results_df[[f'Accuracy', f'Actual {self.label}']].copy().groupby(f'Actual {self.label}').mean()

        return results_df, accuracy_summary

    def mlogit(self):
        df = self.df.copy()
        pca_df, explained_variance = self.pca(n_components=8, df=df)
        print(f"Explained variance: {explained_variance}")
        
        # Separating features and target variable
        X = pca_df.drop(self.label, axis=1)
        y = pca_df[self.label]

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Creating and training the logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)

        # Optionally, evaluate the model
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Return the trained model
        return model
    
    def mlogit_full_data(self):
        df = self.df.copy()
        pca_df, explained_variance, pca = self.pca(n_components=8, df=df, return_pca=True)  # Adjust PCA function to return PCA object
        print(f"Explained variance: {explained_variance}")
        
        # Separating features and target variable
        X = pca_df.drop(self.label, axis=1)
        y = pca_df[self.label]

        # Creating and training the logistic regression model on the entire dataset
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000) # Adjusted max_iter if convergence issues
        model.fit(X, y)

        # Optionally, if you want to evaluate the model internally, you could still predict and evaluate
        # But since you mentioned you've done LOOCV already, you might skip this.
        # If you choose to evaluate:
        y_pred = model.predict(X)
        print("Accuracy (on training data):", accuracy_score(y, y_pred))
        print("Confusion Matrix (on training data):\n", confusion_matrix(y, y_pred))
        print("Classification Report (on training data):\n", classification_report(y, y_pred))

        # Save the model and PCA components
        self.save_components(model, pca)
        # Return the trained model on the entire dataset
        return model

    def lda_loocv(self):
        df = self.df.copy()
        print("loading data...")
        df['image'] = df['path'].apply(lambda x: nh.load(x).normalize_to_quantile().data)
        print("data is loaded and normalized")
        flattened_arrays = df['image'].apply(self.mask_and_flatten)
        data_matrix = np.vstack(flattened_arrays)
        class_labels = df[self.label].values

        loo = LeaveOneOut()
        accuracies = []
        results = []
        print("beginning leave-one-out cross validation for LDA...")
        for train_index, test_index in loo.split(data_matrix):
            print("iteration {} of {}".format(test_index[0]+1, len(data_matrix)))
            X_train, X_test = data_matrix[train_index], data_matrix[test_index]
            y_train, y_test = class_labels[train_index], class_labels[test_index]

            lda = LDA()
            lda.fit(X_train, y_train)
            y_pred = lda.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # Print accuracy for this iteration, and its symptom (class)
            print("accuracy: {}, class: {}".format(accuracy, y_test[0]))
            accuracies.append(accuracy)
            row = {
                f'Actual {self.label}': y_test[0],
                f'Predicted {self.label}': y_pred[0],
                'Accuracy': accuracy,
                'Classification Report': classification_report(y_test, y_pred, output_dict=True)
            }
            results.append(row)

        # Average accuracy
        results_df = pd.DataFrame(results)
        return accuracies, results_df 
    
    def lda(self):
        def unflatten_to_volumetric(coefficients, mask_filepath=f"{current_dir}/MNI152_T1_2mm_brain_mask.nii.gz"):
            mask = nib.load(mask_filepath).get_fdata()
            volumetric_image = np.zeros(mask.shape)
            volumetric_image[mask == 1] = coefficients
            return volumetric_image

        df = self.df.copy()
        df[self.label] = df['path'].apply(lambda x: nh.load(x).normalize_to_quantile().data)  # Assuming NiftiHandler(x).data returns the 3D array
        # Extract and flatten the 3D arrays
        # Instead of list comprehension, let's use pandas apply
        flattened_arrays = df['image'].apply(self.mask_and_flatten)
        # Stack the arrays
        data_matrix = np.vstack(flattened_arrays)

        # Extract class labels (symptoms)
        class_labels = df[self.label].values

        # Apply LDA
        lda = LDA(n_components=2)  # Assuming we want 2 components
        transformed_data = lda.fit_transform(data_matrix, class_labels)
        
        # Extract the coefficients (loadings) for each voxel
        lda_coefficients = lda.coef_

        volumetric_weights = unflatten_to_volumetric(lda_coefficients[0])  # or lda_coefficients[1] for the second discriminant

        # Eigenvalues (or 'explained variance')
        eigenvalues = lda.explained_variance_ratio_

        # Combine LDA results with symptoms for analysis
        lda_df = pd.DataFrame(transformed_data, columns=['LD1', 'LD2'])
        lda_df[self.label] = class_labels
        
        return lda_df, eigenvalues, volumetric_weights    
    
    def save_components(self, model, pca, model_path='model.joblib', pca_path='pca.joblib'):
        """
        Saves the logistic regression model and PCA object to disk.

        :param model: Trained logistic regression model.
        :param pca: Trained PCA object.
        :param model_path: Path to save the logistic regression model.
        :param pca_path: Path to save the PCA object.
        """
        dump(model, model_path)
        dump(pca, pca_path)
        print(f"Model saved to {model_path}")
        print(f"PCA saved to {pca_path}")

    def pca(self, n_components=2, df=None, return_pca=False):
        if df is None:
            df = self.df.copy()
        df['image'] = df['path'].apply(lambda x: nh.load(x).normalize_to_quantile().data)  # Assuming NiftiHandler(x).data returns the 3D array
        # Extract and flatten the 3D arrays
        flattened_arrays = df['image'].apply(self.mask_and_flatten)
        # Stack the arrays
        data_matrix = np.vstack(flattened_arrays)

        # Apply PCA with n_components
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data_matrix)

        # Get the explained variance
        explained_variance = pca.explained_variance_ratio_

        # Dynamic column names based on number of components
        column_names = [f'PC{i+1}' for i in range(n_components)]

        # Combine PCA results with symptoms for analysis
        pca_df = pd.DataFrame(transformed_data, columns=column_names)
        pca_df[self.label] = df[self.label].values
        
        if return_pca:
            return pca_df, explained_variance, pca
        return pca_df, explained_variance

    def cross_correlation_analysis(self):
        """
        Performs a cross-correlation analysis on image data stored in the class instance.
        Generates two DataFrames:
        1. A DataFrame of pairwise correlations between images.
        2. A grouped DataFrame summarizing correlations by symptom pairs.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two DataFrames, the first being pairwise 
        image correlations and the second being symptom-wise average correlations.
        """
        def mean_confidence_interval_correlation(data, confidence=0.95):
            mean = data.mean()
            n = len(data)
            stderr = stats.sem(data)  # Standard error of the mean
            h = stderr * stats.t.ppf((1 + confidence) / 2, n - 1)  # Margin of error
            lower_bound = max(-1, mean - h)
            upper_bound = min(1, mean + h)
            return mean, lower_bound, upper_bound
            
        # Create a DataFrame with images loaded
        df = self.df.copy()
        df['image'] = df['path'].apply(lambda x: nh.load(x))

        # Initialize an empty DataFrame for correlations
        paths = df['path']
        correlation_df = pd.DataFrame(index=paths, columns=paths)

        # Fill the diagonal with 1s (self-correlation)
        np.fill_diagonal(correlation_df.values, 1.0)

        # Calculate correlations for unique pairs
        for (idx1, row1), (idx2, row2) in itertools.combinations(df.iterrows(), 2):
            path1, image1 = row1['path'], row1['image']
            path2, image2 = row2['path'], row2['image']
            correlation = ImageComparison().correlate_images(image1, image2)

            correlation_df.at[path1, path2] = correlation
            correlation_df.at[path2, path1] = correlation

     
        # Transform correlation data into a symptom-pair level summary
        symptom_correlation_df = self._create_symptom_level_correlation_df(df, correlation_df)

        # Group by symptom pairs and calculate the mean correlation
        symptom_correlation_df['correlation'] = pd.to_numeric(symptom_correlation_df['correlation'], errors='coerce')
        grouped_symptom_corr = symptom_correlation_df.groupby([f'{self.label}1', f'{self.label}2'])['correlation'].apply(mean_confidence_interval_correlation).reset_index()
        grouped_symptom_corr[['mean_correlation', 'lower_95_CI', 'upper_95_CI']] = pd.DataFrame(grouped_symptom_corr['correlation'].tolist(), index=grouped_symptom_corr.index)
        grouped_symptom_corr.drop(columns=['correlation'], inplace=True)

        return symptom_correlation_df, grouped_symptom_corr

    def _create_symptom_level_correlation_df(self, df, correlation_df):
        """
        Transforms a correlation DataFrame into a symptom-level summary.

        Parameters:
        df (pd.DataFrame): The original DataFrame with image paths and symptoms.
        correlation_df (pd.DataFrame): DataFrame containing pairwise correlations between images.

        Returns:
        pd.DataFrame: A DataFrame summarizing correlations at the symptom-pair level.
        """
        # Rename the index of correlation_df to avoid conflict with 'path' column
        renamed_index = ["{}_{}".format(path, i) for i, path in enumerate(correlation_df.index)]
        correlation_df.index = renamed_index
        correlation_df.columns = renamed_index

        # Flatten the correlation matrix and reset the index
        symptom_corr_flat = correlation_df.where(np.triu(np.ones(correlation_df.shape), k=1).astype(bool)).stack().reset_index()
        symptom_corr_flat.rename(columns={'level_0': 'id1', 'level_1': 'id2', 0: 'correlation'}, inplace=True)

        # Map the renamed index back to original path and symptom
        map_to_original = dict(zip(renamed_index, df['path']))
        symptom_corr_flat['path1'] = symptom_corr_flat['id1'].map(map_to_original)
        symptom_corr_flat['path2'] = symptom_corr_flat['id2'].map(map_to_original)
        symptom_corr_flat[f'{self.label}1'] = symptom_corr_flat['path1'].map(df.set_index('path')[self.label])
        symptom_corr_flat[f'{self.label}2'] = symptom_corr_flat['path2'].map(df.set_index('path')[self.label])

        # Drop unnecessary columns and rows where correlation is 1
        symptom_corr_flat.drop(columns=['id1', 'id2'], inplace=True)
        symptom_corr_flat = symptom_corr_flat[symptom_corr_flat['correlation'] != 1]

        return symptom_corr_flat
    
    def k_means_clustering(self, df=None, n_clusters=6):
        """
        Perform k-means clustering on a dataframe containing multidimensional numpy arrays.

        :param dataframe: DataFrame containing the data.
        :param df: The dataframe, which includes a column containing numpy arrays.
        :param n_clusters: Number of clusters to form.
        :return: The input dataframe with an additional column 'cluster' indicating the cluster each row belongs to.
        """
        if df is None:
            df = self.df.copy()
        
        df['data'] = df['path'].apply(lambda x: nh.load(x).normalize_to_quantile().data).apply(self.mask_and_flatten)
        # df['shape'] = df['data'].apply(lambda x: x.shape)
        # display(df)
        # data = df['data']
        data_array = np.vstack(df['data'].values)
        # display(data) # Maybe I need to do a vstack or something here?
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_array)

        # Add the cluster information to the original dataframe
        df['cluster'] = clusters
        return df

    def correlate_and_classify_coor(self, data_dict):
        if data_dict is None:
            data_dict = self.df.copy()

        # Preprocess data outside the loop
        data_dict['path'] = data_dict['path'].apply(lambda x: nh.load(x).normalize_to_quantile())
        print("data is loaded and normalized")

        # Function to be applied in parallel for each row
        def process_row(row, data_dict):
            print("working on row {} of {}".format(row.name + 1, len(data_dict)))
            original_image_map = row['path']
            actual_symptom = row[self.label]
            comparer = ImageComparison()

            # Exclude the current row for comparison
            mask = data_dict.index != row.name
            data_dict_without_idx = data_dict.loc[mask]

            # Perform the comparison
            description, df_corr = comparer.correlate_with_symptoms(original_image_map, data_dict_without_idx)
            
            p_value = float(description.split('=')[1].split(')')[0].strip())
            percentage_of_comparisons_that_are_significant = df_corr['percentage_of_comparisons_that_are_significant'].iloc[0]
            symptoms_with_significant_differences = df_corr[f'{self.label}s_with_significant_differences'].iloc[0]
            # print(symptoms_with_significant_differences)
            if symptoms_with_significant_differences:
                p_value_against_next_best_symptoms = max(symptoms_with_significant_differences.values())
            else: 
                p_value_against_next_best_symptoms = None

            # Ensure df_corr has at least one row and contains 'symptom' column
            if not df_corr.empty and self.label in df_corr.columns:
                predicted_symptom = df_corr[self.label].iloc[0]
                correct_prediction = actual_symptom == predicted_symptom
                print(f"Actual {self.label}: {actual_symptom}, predicted {self.label}: {predicted_symptom}")
            else:
                predicted_symptom = None
                correct_prediction = False
                print(f"Actual {self.label}: {actual_symptom}, predicted {self.label}: Not Available")

            # Summarize results
            row = {
                'description': description,
                f'actual_{self.label}': actual_symptom,
                f'predicted_{self.label}': predicted_symptom,
                'correct_prediction': correct_prediction,
                'p_value': p_value,
                'percentage_of_comparisons_that_are_significant': percentage_of_comparisons_that_are_significant,
                f'{self.label}s_with_significant_differences': symptoms_with_significant_differences,
                f'p_value_against_next_best_{self.label}s': p_value_against_next_best_symptoms
            }
            return row
        
        results = [process_row(row, data_dict) for idx, row in data_dict.iterrows()]
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df