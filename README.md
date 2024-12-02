# PCA on Face Images

In this assignment, I implemented **Principal Component Analysis (PCA)** on pictures of faces for dimensionality reduction and visualization.

## Description

The task was to implement PCA from scratch (without using libraries such as Sklearn's PCA) and apply it to a dataset of face images. Afterward, I ran experiments to explore the effect of PCA in reducing the dimensionality of facial image data.

### Steps:
1. **PCA Implementation**: 
   - Implemented PCA manually (without using Sklearn or any other library that implements PCA).
   
2. **Data Selection**: 
   - Chose a random person from the dataset with more than 50 pictures available.
   - Implemented a utility function to load pictures of that specific person.

3. **Matrix Construction**:
   - Constructed a matrix **X** where each row represents a flattened version of a picture of the selected person.

4. **Experiments**:
   - **Experiment 1**: Ran PCA on the matrix **X** with `k = 10` and plotted the first 10 eigenvectors as images.
   - **Experiment 2**: Ran PCA for different values of `k = 1, 5, 10, 30, 50, 100` and reduced the dimensionality of the images to each of these values. Then, I selected 5 random pictures (the same 5 for all values of `k`) and plotted the original images alongside the reconstructed images obtained by transforming the reduced representations back to the original dimension. I also plotted, as a function of `k`, the sum of the ℓ2 distances between the original and reconstructed images for the 5 selected pictures.


