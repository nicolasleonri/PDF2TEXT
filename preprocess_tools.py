import os
import cv2
from PIL import Image
import numpy as np
import imutils

"""
TODOS:
1. Test different parameters
2. Add corrections based on literature
3. Automatically determine best parameters to be used
"""


def get_imlist(path):
    """
    Returns a list of filenames for all jpg images in a directory.

    Parameters:
    path (str): Path to the directory to search for jpg images.

    Returns:
    list: A list of filenames for all jpg images in the specified directory.
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]


def get_fullpath(dir_name, base_filename):
    """
    This function returns the full path of a file, by combining the directory name and the base file name.

    Parameters:
    dir_name: the name of the directory (string)
    base_filename: the name of the file (string)

    Returns:
    The full path of the file (string)
    """
    return os.path.join(dir_name, base_filename)


def show_image_from_path(path):
    """
    Creates a window and displays an image in it.

    Parameters:
    image: an image represented as an array.

    Details:
    The function creates a window using the cv2.namedWindow function with a specified window name and the option to adjust the window size using the cv2.WINDOW_NORMAL argument.
    The image is then displayed in the window using cv2.imshow.
    The function waits for a keyboard event with cv2.waitKey and destroys all windows when it is done with cv2.destroyAllWindows.
    """
    name_of_window = 'Test_Window'
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)

    image = cv2.imread(path)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_from_variable(input):
    name_of_window = 'Test_Window'
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)

    cv2.imshow(name_of_window, input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(image):
    def normalize(image):
        """
        This function normalizes an image by converting its values to the range [0, 255].

        Parameters:
        image (ndarray): The input image to be normalized.

        Returns:
        ndarray: The normalized image.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        # Create a zero array with the same shape as the input image
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        # Normalize the image
        output = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        return output

    def scale(image, factor=2.0):
        """
        Function to scale an image using OpenCV.

        Parameters:
        image (numpy.ndarray): The image to be scaled, in BGR format.
        factor (float): The scaling factor to be applied to the image. Default value is 2.0.

        Returns:
        numpy.ndarray: The scaled image in RGB format.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # fx and fy specify the scaling factor
        fx = factor
        fy = factor
        image_scaled = cv2.resize(
            image, (int(width * fx), int(height * fy)), interpolation=cv2.INTER_CUBIC)
        output = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
        return output

    def denoise_color(image):
        """
        Function to denoise a colored image.

        Parameters:
            image: Input colored image.

        Returns:
            output: Denoised image.

        Details:
        The parameters in cv2.fastNlMeansDenoisingColored function in the given code are:
        image: The input image on which denoising operation needs to be performed.
        None: This parameter is for mask image. If you pass None, then the function considers all pixels to be used for denoising operation.
        3: This parameter represents the search window size used to find the similar pixels. The larger the search window size, the longer the function will take to complete the operation.
        3: This parameter represents the size of the neighborhood area to consider for finding the average color for denoising. Larger neighborhood size means larger averaging area and greater blurring effect.
        7: The parameter represents the standard deviation of the Gaussian filter used to smooth the image prior to denoising. A larger standard deviation results in greater smoothing.
        21: The parameter is the standard deviation of the Gaussian filter applied to the image after denoising is complete. The purpose of this filter is to adjust the brightness and contrast of the denoised image. A larger standard deviation results in greater brightness and contrast correction.
        """
        output = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        return output

    def remove_pictures(image, lower_area=15000, upper_area=35000):
        # Convert image from BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to the grayscale image to reduce noise
        # (3, 3) is the size of the kernel (square) used for blurring
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply Canny edge detection to the blurred image
        # Lower threshold of 120 and upper threshold of 255
        # "1" is the aperture size used to find edges
        canny = cv2.Canny(blur, 120, 255, 1)

        # Create a structuring element for morphological operations
        # (3,3) is the size of the square used for dilation and erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Close small holes in the image using morphological operations
        # iterations=2 specifies the number of times the closing operation is applied
        close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the closed image
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Loop through all contours
        for c in cnts:
            # Calculate area of each contour
            area = cv2.contourArea(c)

            # Draw rectangles around contours with area between lower_area and upper_area
            if area > lower_area and area < upper_area:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (255, 255, 255), -1)

        return image

    def thresh(image):
        """Applies Adaptive Threshold to the input image

        Parameters:
        image (np.ndarray): Input image in BGR format

        Returns:
        np.ndarray: Output image after applying Adaptive Threshold

        Parameters Explanation:
        1. img_grey: Input image in grayscale
        2. 255: The maximum pixel value to be used with the THRESH_BINARY thresholding type.
        3. cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Adaptive Thresholding Algorithm used. Option computes the threshold for smaller regions.
        4. cv2.THRESH_BINARY: The type of thresholding to be applied. cv2.THRESH_BINARY indicates binary thresholding, which sets all pixels above the threshold to maxVal and the rest to 0.
        5. 11: The size of a pixel neighborhood used to calculate the threshold value.
        6. 2: Constant subtracted from the mean or weighted mean. The calculated threshold value is the mean minus this constant.
        """
        img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = cv2.adaptiveThreshold(
            img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return output

    def denoise_and_threshold(input_image):
        """
        Denoise and apply thresholding to the input image.

        Parameters:
            input_image (np.ndarray): The input image in BGR color space.

        Returns:
            np.ndarray: The denoised and thresholded image in BGR color space.

        Explanation:
            The function denoises the input image using morphological operations and then applies Otsu thresholding to separate the object from the background.
            The image is split into its blue, green, and red channels and Otsu thresholding is applied to each channel separately.
            The denoised and thresholded image is then returned in BGR color space.
        """
        # Create a copy of the input image
        denoised_image = input_image.copy()

        # Define a morphological filter with a (1, 1) kernel and apply morphological closing and opening
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        denoised_image = cv2.morphologyEx(
            denoised_image, cv2.MORPH_CLOSE, morph_kernel)
        denoised_image = cv2.morphologyEx(
            denoised_image, cv2.MORPH_OPEN, morph_kernel)

        # Split the denoised image into its blue, green, and red channels
        b, g, r = cv2.split(denoised_image)

        # Apply Otsu thresholding to each channel
        _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Merge the channels back together
        denoised_and_thresholded = cv2.merge((b, g, r))

        return denoised_and_thresholded

    # Invertion:
    def invert(input):
        output = (255-input)
        return output

    return invert(thresh(denoise_and_threshold(scale((denoise_color(normalize(image)))))))
    # return scale(denoise_color(normalize(image)))


def remove_images(image):

    # Threshold reduction:
    def thresh_reduction(input, x, y, thresh_value, iterations=1):
        # Closing on X x Y
        # Thresh value is int
        ret, thresh = cv2.threshold(input, int(
            thresh_value), 255, cv2.THRESH_BINARY_INV)
        # ret, thresh = cv2.threshold(input, int(thresh_value), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # kernel = np.ones((int(x), int(y)),np.uint8)
        # closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel, iterations = int(iterations))
        return thresh

    # Structural opening:
    def structural_opening(input, x, y):
        kernel = np.ones((x, y), np.uint8)
        output = cv2.morphologyEx(input, cv2.MORPH_OPEN, kernel)
        return output

    # Dilation:
    def dilate(input, x, y, iterations=1):
        kernel = np.ones((x, y), np.uint8)
        output = cv2.dilate(input, kernel, iterations=int(iterations))
        return output

    # Invertion:
    def invert(input):
        output = (255-input)
        return output

    # Read image:
    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])

    # Convert image from BGR to binary:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    (thresh, binary) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # First step:
    output = thresh_reduction(binary, 4, 1, 1, 2)

    # Morphology operation:
    output = thresh_reduction(output, 4, 1, 4)
    output = thresh_reduction(output, 4, 1, 3)

    output = structural_opening(output, 5, 5)
    output = dilate(output, 1, 4, 2)

    # Last step:
    output = structural_opening(output, 3, 3)
    output = dilate(output, 1, 4, 2)

    # output = invert(output)
    return output


def find_contours(input, lower_area=500, upper_area=10000):

    def is_elipse(c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        return not len(approx) == 4

    def is_square(c):
        epsilon = 0.08*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        return approx

    approx_list = []

    output = (255-input)
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(
        output, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    print("Number of contours detected:", len(contours))

    mask = np.ones(input.shape[:2], dtype="uint8") * 255

    # loop over the contours
    for c in contours:
        area = cv2.contourArea(c)
        print(area)
        print(is_square(c))
        if len(is_square(c)) == 4 and area > 200:
            print(is_square(c))
            approx_list.append(is_square(c))
        # if area > 250 and area <8000:
            # cv2.drawContours(mask, [c], -1, 0, -1)

    # show_image_from_variable(mask)

    # output = cv2.drawContours(input, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    output = cv2.drawContours(input, contours=approx_list, contourIdx=-1,
                              color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return output


#image_path = get_fullpath(os.getcwd(), "test4.png")

#show_image_from_path(image_path)

#image = preprocess(image_path)
#show_image_from_variable(image)

# image = remove_images(image)
# show_image_from_variable(image)

# image = find_contours(image)
# show_image_from_variable(image)
