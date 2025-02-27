import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Path to the CSV file
file_path = "Path/to/yo/file"

# Load the data
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
    print(df.head())  # Check the first few rows of the DataFrame
except Exception as e:
    print(f"Error loading file: {e}")
    df = None

if df is not None:
    # Set the index to the 'id' (assuming the first column is 'id')
    df.set_index('id', inplace=True)

    # Log Transformation (add a small constant to avoid log(0))
    #transformed_df = np.log1p(df)
   
   # Square Root Transformation
    #transformed_df = np.sqrt(df)
   
    
# Box-Cox Transformation (only for positive data)
    from scipy.stats import boxcox   
    transformed_df = df.apply(lambda x: boxcox(x + 1)[0])

    

# Yeo-Johnson Transformation
    #from sklearn.preprocessing import PowerTransformer
    #transformer = PowerTransformer(method='yeo-johnson')
    #transformed_data = transformer.fit_transform(df)
    #transformed_df = pd.DataFrame(transformed_data, index=df.index, columns=df.columns)



    # Standardizing the transformed data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transformed_df.T)  # Transpose to standardize samples
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Creating a DataFrame with PCA results
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df.columns)
    
    # Plotting the PCA results
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    
    for sample in pca_df.index:
        plt.text(pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2'], sample)
    
    plt.title('PCA of Pre_filtered_mirnas')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()



    # Performing PCA with 3 components for 3D plot
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)

    # DataFrame creation
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'], index=df.columns)

    # Plotting results
    fig = plt.figure(figsize=(18, 8))
    
#    # 3D PCA results
#    ax = fig.add_subplot(121, projection='3d')
#    ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c='blue', label='Samples')
#for sample in pca_df.index:
#    ax.text(pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2'], pca_df.loc[sample, 'PC3'], sample)
#        
#    ax.set_title('PCA of Log Transformed Gene Expression Data')
#    ax.set_xlabel('Principal Component 1')
#    ax.set_ylabel('Principal Component 2')
#    ax.set_zlabel('Principal Component 3')
#    ax.legend()

    # 2D density plot
    ax = fig.add_subplot(122)
    sns.kdeplot(x=pca_df['PC1'], y=pca_df['PC2'], cmap='Blues', shade=True, fill=True)

    for sample in pca_df.index:
        ax.text(pca_df.loc[sample, 'PC1'], pca_df.loc[sample, 'PC2'], sample)

    ax.set_title('Density Plot of PCA (PC1 vs PC2, Log Transformed Data)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()

    explained_variance = pca.explained_variance_ratio_
    print(f'Explained variance by components: {explained_variance}')

    # Scree plot
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot of PCA (First 3 Components, Log Transformed Data)')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    print("PCA analysis completed")
else:
    print("Data loading failed. Please check the file path and format.")
