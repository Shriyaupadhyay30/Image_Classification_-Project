#Checking one image's pixel intensity and edge detection to see if its good

#Picking a random image from the DataFrame
ran_index = random.choice(df.index)
ran_filepath = df.loc[ran_index, 'filepaths']
ran_label = df.loc[ran_index, 'labels']

#Loading the selected image in grayscale
img = cv2.imread(ran_filepath, cv2.IMREAD_GRAYSCALE)

#Checking if the image was loaded properly
if img is not None:
    print(f"Selected Image: {ran_filepath}, Label: {ran_label}")
    
    #Pixel Intensity Distribution (Histogram) plot
    plt.figure(figsize=(10, 6))

    #Flatting the image array
    plt.hist(img.ravel(), bins=256, color='#6A8D73', alpha=0.7) 
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

    #Basic Statistics of Pixel Intensities
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    min_intensity = np.min(img)
    max_intensity = np.max(img)
    print(f"Image Statistics - Mean: {mean_intensity}, Standard Deviation: {std_intensity}, Min: {min_intensity}, Max: {max_intensity}")
    
    #Displaying the  Grey scale Image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Image - Label: {ran_label}")
    plt.axis('off')
    plt.show()

    #Edge Detection Using Sobel Filter from opencv
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    #Visualizing the Edge Detected Image
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap='hot')
    plt.title(f"Edge Detection with Sobel for Image - Label: {ran_label}")
    plt.axis('off')
    plt.show()

else:
    print("Error loading the image!")
