# # import string
# # import easyocr

# # # Initialize the OCR reader
# # reader = easyocr.Reader(['en'], gpu=False)

# # # Mapping dictionaries for character conversion
# # dict_char_to_int = {'O': '0',
# #                     'I': '1',
# #                     'J': '3',
# #                     'A': '4',
# #                     'G': '6',
# #                     'S': '5'}

# # dict_int_to_char = {'0': 'O',
# #                     '1': 'I',
# #                     '3': 'J',
# #                     '4': 'A',
# #                     '6': 'G',
# #                     '5': 'S'}


# # def write_csv(results, output_path):
# #     """
# #     Write the results to a CSV file.

# #     Args:
# #         results (dict): Dictionary containing the results.
# #         output_path (str): Path to the output CSV file.
# #     """
# #     with open(output_path, 'w') as f:
# #         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
# #                                                 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
# #                                                 'license_number_score'))

# #         for frame_nmr in results.keys():
# #             for car_id in results[frame_nmr].keys():
# #                 print(results[frame_nmr][car_id])
# #                 if 'car' in results[frame_nmr][car_id].keys() and \
# #                    'license_plate' in results[frame_nmr][car_id].keys() and \
# #                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
# #                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
# #                                                             car_id,
# #                                                             '[{} {} {} {}]'.format(
# #                                                                 results[frame_nmr][car_id]['car']['bbox'][0],
# #                                                                 results[frame_nmr][car_id]['car']['bbox'][1],
# #                                                                 results[frame_nmr][car_id]['car']['bbox'][2],
# #                                                                 results[frame_nmr][car_id]['car']['bbox'][3]),
# #                                                             '[{} {} {} {}]'.format(
# #                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][0],
# #                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][1],
# #                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][2],
# #                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][3]),
# #                                                             results[frame_nmr][car_id]['license_plate']['bbox_score'],
# #                                                             results[frame_nmr][car_id]['license_plate']['text'],
# #                                                             results[frame_nmr][car_id]['license_plate']['text_score'])
# #                             )
# #         f.close()


# # def license_complies_format(text):
# #     """
# #     Check if the license plate text complies with the required format.

# #     Args:
# #         text (str): License plate text.

# #     Returns:
# #         bool: True if the license plate complies with the format, False otherwise.
# #     """
# #     if len(text) != 7:
# #         return False

# #     if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
# #        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
# #        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
# #        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
# #        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
# #        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
# #        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
# #         return True
# #     else:
# #         return False


# # def format_license(text):
# #     """
# #     Format the license plate text by converting characters using the mapping dictionaries.

# #     Args:
# #         text (str): License plate text.

# #     Returns:
# #         str: Formatted license plate text.
# #     """
# #     license_plate_ = ''
# #     mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
# #                2: dict_char_to_int, 3: dict_char_to_int}
# #     for j in [0, 1, 2, 3, 4, 5, 6]:
# #         if text[j] in mapping[j].keys():
# #             license_plate_ += mapping[j][text[j]]
# #         else:
# #             license_plate_ += text[j]

# #     return license_plate_


# # def read_license_plate(license_plate_crop):
# #     """
# #     Read the license plate text from the given cropped image.

# #     Args:
# #         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

# #     Returns:
# #         tuple: Tuple containing the formatted license plate text and its confidence score.
# #     """

# #     detections = reader.readtext(license_plate_crop)

# #     for detection in detections:
# #         bbox, text, score = detection

# #         text = text.upper().replace(' ', '')

# #         if license_complies_format(text):
# #             return format_license(text), score

# #     return None, None


# # def get_car(license_plate, vehicle_track_ids):
# #     """
# #     Retrieve the vehicle coordinates and ID based on the license plate coordinates.

# #     Args:
# #         license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
# #         vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

# #     Returns:
# #         tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
# #     """
# #     x1, y1, x2, y2, score, class_id = license_plate

# #     foundIt = False
# #     for j in range(len(vehicle_track_ids)):
# #         xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

# #         if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
# #             car_indx = j
# #             foundIt = True
# #             break

# #     if foundIt:
# #         return vehicle_track_ids[car_indx]

# #     return -1, -1, -1, -1, -1



# #----------------------------updated version-------------------------

# import string
# import easyocr
# import cv2
# import numpy as np
# import re

# # Initialize the OCR reader with better settings
# reader = easyocr.Reader(['en'], gpu=False)

# # Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'B': '8'}
# dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '8': 'B'}

# def preprocess_license_plate(image):
#     """
#     Enhanced preprocessing for license plate images
#     """
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Resize image if too small
#     height, width = image.shape
#     if height < 50 or width < 150:
#         scale_factor = max(50/height, 150/width)
#         new_width = int(width * scale_factor)
#         new_height = int(height * scale_factor)
#         image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
#     # Apply different preprocessing techniques
#     processed_images = []
    
#     # 1. Original with denoising
#     denoised = cv2.fastNlMeansDenoising(image)
#     processed_images.append(("denoised", denoised))
    
#     # 2. Histogram equalization
#     equalized = cv2.equalizeHist(image)
#     processed_images.append(("equalized", equalized))
    
#     # 3. Adaptive thresholding
#     adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     processed_images.append(("adaptive_thresh", adaptive))
    
#     # 4. OTSU thresholding
#     _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     processed_images.append(("otsu", otsu))
    
#     # 5. Morphological operations to clean up
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
#     processed_images.append(("morphological", morph))
    
#     # 6. Gaussian blur + threshold
#     blurred = cv2.GaussianBlur(image, (3, 3), 0)
#     _, thresh_blur = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
#     processed_images.append(("gaussian_thresh", thresh_blur))
    
#     return processed_images

# def clean_text(text):
#     """
#     Clean and standardize OCR output
#     """
#     if not text:
#         return ""
    
#     # Remove spaces and convert to uppercase
#     text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(',', '')
    
#     # Remove non-alphanumeric characters except common license plate characters
#     text = re.sub(r'[^A-Z0-9]', '', text)
    
#     return text

# def validate_license_format(text):
#     """
#     Check if text looks like a valid license plate
#     """
#     if not text or len(text) < 3:
#         return False
    
#     # Must contain both letters and numbers for most license plates
#     has_letters = bool(re.search(r'[A-Z]', text))
#     has_numbers = bool(re.search(r'[0-9]', text))
    
#     # Length should be reasonable (3-8 characters is typical)
#     if len(text) < 3 or len(text) > 8:
#         return False
    
#     return has_letters or has_numbers  # Allow plates with only letters or only numbers

# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file.
#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score'))

#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 print(results[frame_nmr][car_id])
#                 if 'car' in results[frame_nmr][car_id].keys() and \
#                    'license_plate' in results[frame_nmr][car_id].keys() and \
#                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
#                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                           car_id,
#                                                           '[{} {} {} {}]'.format(
#                                                               results[frame_nmr][car_id]['car']['bbox'][0],
#                                                               results[frame_nmr][car_id]['car']['bbox'][1],
#                                                               results[frame_nmr][car_id]['car']['bbox'][2],
#                                                               results[frame_nmr][car_id]['car']['bbox'][3]),
#                                                           '[{} {} {} {}]'.format(
#                                                               results[frame_nmr][car_id]['license_plate']['bbox'][0],
#                                                               results[frame_nmr][car_id]['license_plate']['bbox'][1],
#                                                               results[frame_nmr][car_id]['license_plate']['bbox'][2],
#                                                               results[frame_nmr][car_id]['license_plate']['bbox'][3]),
#                                                           results[frame_nmr][car_id]['license_plate']['bbox_score'],
#                                                           results[frame_nmr][car_id]['license_plate']['text'],
#                                                           results[frame_nmr][car_id]['license_plate']['text_score'])
#                            )

# def license_complies_format(text):
#     """
#     Check if the license plate text complies with the required format.
#     This is a more flexible version that works with various license plate formats.
#     """
#     if not text or len(text) < 3:
#         return False
    
#     # Basic validation - must be alphanumeric and reasonable length
#     if not text.isalnum():
#         return False
    
#     if len(text) > 8:
#         return False
    
#     return True

# def format_license(text):
#     """
#     Format the license plate text by converting characters using the mapping dictionaries.
#     This version is more flexible and works with different license plate formats.
#     """
#     if not text:
#         return text
    
#     license_plate_ = ''
    
#     for i, char in enumerate(text):
#         # Apply character corrections based on context
#         if char.isdigit():
#             # If it's a number but looks like it should be a letter
#             if char in dict_int_to_char:
#                 # Only convert if we're in a position that typically has letters
#                 # This is format-agnostic
#                 license_plate_ += dict_int_to_char.get(char, char)
#             else:
#                 license_plate_ += char
#         else:
#             # If it's a letter but looks like it should be a number
#             if char in dict_char_to_int:
#                 # Only convert if we're in a position that typically has numbers
#                 license_plate_ += dict_char_to_int.get(char, char)
#             else:
#                 license_plate_ += char
    
#     return license_plate_

# def read_license_plate(license_plate_crop):
#     """
#     Enhanced license plate text reading with multiple preprocessing attempts
#     """
#     if license_plate_crop is None or license_plate_crop.size == 0:
#         return None, None
    
#     # Get multiple preprocessed versions
#     processed_images = preprocess_license_plate(license_plate_crop)
    
#     best_text = None
#     best_score = 0
#     best_method = None
#     all_results = []
    
#     # Try OCR on each preprocessed version
#     for method_name, processed_img in processed_images:
#         try:
#             # Use EasyOCR with optimized settings for license plates
#             detections = reader.readtext(
#                 processed_img,
#                 width_ths=0.4,      # Adjust width threshold
#                 height_ths=0.4,     # Adjust height threshold
#                 paragraph=False,     # Don't group text
#                 detail=1            # Return detailed results
#             )
            
#             for detection in detections:
#                 if len(detection) >= 3:
#                     bbox, text, score = detection
                    
#                     # Clean the text
#                     cleaned_text = clean_text(text)
                    
#                     if cleaned_text and len(cleaned_text) >= 2:  # Minimum length
#                         # Validate if it looks like a license plate
#                         if validate_license_format(cleaned_text):
#                             all_results.append((cleaned_text, score, method_name))
                            
#                             print(f"    OCR ({method_name}): '{cleaned_text}' (raw: '{text}') score: {score:.3f}")
                            
#                             if score > best_score:
#                                 best_text = cleaned_text
#                                 best_score = score
#                                 best_method = method_name
        
#         except Exception as e:
#             print(f"    OCR error with {method_name}: {e}")
#             continue
    
#     # If we found valid text, apply formatting
#     if best_text and best_score > 0.3:  # Lower threshold for debugging
#         # Try to format the license plate
#         formatted_text = format_license(best_text)
        
#         # Use formatted version if it still looks valid
#         if license_complies_format(formatted_text):
#             print(f"    Formatted: '{best_text}' -> '{formatted_text}'")
#             return formatted_text, best_score
#         else:
#             # Use original if formatting made it invalid
#             return best_text, best_score
    
#     # If no good results, return the best attempt even if low confidence
#     if all_results:
#         # Sort by score and return best
#         all_results.sort(key=lambda x: x[1], reverse=True)
#         best_attempt = all_results[0]
#         print(f"    Best attempt: '{best_attempt[0]}' score: {best_attempt[1]:.3f}")
#         return best_attempt[0], best_attempt[1]
    
#     return None, None

# def get_car(license_plate, vehicle_track_ids):
#     """
#     Retrieve the vehicle coordinates and ID based on the license plate coordinates.
#     Args:
#         license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
#         vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
#     Returns:
#         tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
#     """
#     x1, y1, x2, y2, score, class_id = license_plate
    
#     foundIt = False
#     best_match = None
#     best_overlap = 0
    
#     for j in range(len(vehicle_track_ids)):
#         xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        
#         # Check if license plate is inside vehicle bounding box
#         if x1 >= xcar1 and y1 >= ycar1 and x2 <= xcar2 and y2 <= ycar2:
#             # Calculate overlap area for better matching
#             overlap_area = (min(x2, xcar2) - max(x1, xcar1)) * (min(y2, ycar2) - max(y1, ycar1))
            
#             if overlap_area > best_overlap:
#                 best_overlap = overlap_area
#                 best_match = j
#                 foundIt = True
    
#     if foundIt:
#         return vehicle_track_ids[best_match]
    
#     return -1, -1, -1, -1, -1

import string
import easyocr
import cv2
import numpy as np
import re

# Initialize the OCR reader with better settings
reader = easyocr.Reader(['en'], gpu=False)

# CORRECTED: Fixed mapping dictionaries - only map common OCR errors
# These should only correct obvious OCR misreads, not valid characters
dict_char_to_int = {
    'O': '0',  # Letter O often misread as number 0
    'I': '1',  # Letter I often misread as number 1
    'S': '5',  # Letter S sometimes misread as number 5
    'Z': '2',  # Letter Z sometimes misread as number 2
    'B': '8',  # Letter B sometimes misread as number 8
}

dict_int_to_char = {
    '0': 'O',  # Number 0 often misread as letter O
    '1': 'I',  # Number 1 often misread as letter I
    '5': 'S',  # Number 5 sometimes misread as letter S
    '2': 'Z',  # Number 2 sometimes misread as letter Z
    '8': 'B',  # Number 8 sometimes misread as letter B
}

# REMOVED problematic mappings:
# 'J': '3' - This was causing your 3s to become Js!
# '3': 'J' - This would cause Js to become 3s
# 'A': '4' - This was causing 4s to become As
# '4': 'A' - This would cause As to become 4s
# 'G': '6' - This was causing 6s to become Gs  
# '6': 'G' - This would cause Gs to become 6s

def preprocess_license_plate(image):
    """
    Enhanced preprocessing for license plate images
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image if too small
    height, width = image.shape
    if height < 50 or width < 150:
        scale_factor = max(50/height, 150/width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply different preprocessing techniques
    processed_images = []
    
    # 1. Original with denoising
    denoised = cv2.fastNlMeansDenoising(image)
    processed_images.append(("denoised", denoised))
    
    # 2. Histogram equalization
    equalized = cv2.equalizeHist(image)
    processed_images.append(("equalized", equalized))
    
    # 3. Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_images.append(("adaptive_thresh", adaptive))
    
    # 4. OTSU thresholding
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("otsu", otsu))
    
    # 5. Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    processed_images.append(("morphological", morph))
    
    # 6. Gaussian blur + threshold
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh_blur = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    processed_images.append(("gaussian_thresh", thresh_blur))
    
    return processed_images

def clean_text(text):
    """
    Clean and standardize OCR output
    """
    if not text:
        return ""
    
    # Remove spaces and convert to uppercase
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(',', '')
    
    # Remove non-alphanumeric characters except common license plate characters
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    return text

def validate_license_format(text):
    """
    Check if text looks like a valid license plate
    """
    if not text or len(text) < 3:
        return False
    
    # Must contain both letters and numbers for most license plates
    has_letters = bool(re.search(r'[A-Z]', text))
    has_numbers = bool(re.search(r'[0-9]', text))
    
    # Length should be reasonable (3-8 characters is typical)
    if len(text) < 3 or len(text) > 8:
        return False
    
    return has_letters or has_numbers  # Allow plates with only letters or only numbers

def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                          car_id,
                                                          '[{} {} {} {}]'.format(
                                                              results[frame_nmr][car_id]['car']['bbox'][0],
                                                              results[frame_nmr][car_id]['car']['bbox'][1],
                                                              results[frame_nmr][car_id]['car']['bbox'][2],
                                                              results[frame_nmr][car_id]['car']['bbox'][3]),
                                                          '[{} {} {} {}]'.format(
                                                              results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                              results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                              results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                              results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                          results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                          results[frame_nmr][car_id]['license_plate']['text'],
                                                          results[frame_nmr][car_id]['license_plate']['text_score'])
                           )

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    This is a more flexible version that works with various license plate formats.
    """
    if not text or len(text) < 3:
        return False
    
    # Basic validation - must be alphanumeric and reasonable length
    if not text.isalnum():
        return False
    
    if len(text) > 8:
        return False
    
    return True

def smart_format_license(text):
    """
    CONSERVATIVE formatting - only fix obvious OCR errors
    """
    if not text:
        return text
    
    formatted_text = ""
    
    # Only apply corrections for very obvious OCR mistakes
    for char in text:
        # Apply conservative corrections
        if char in dict_char_to_int:
            # Only convert if it's a very obvious mistake (like O vs 0)
            formatted_text += dict_char_to_int[char]
        elif char in dict_int_to_char:
            # Only convert if it's a very obvious mistake (like 0 vs O)
            formatted_text += dict_int_to_char[char]
        else:
            # Keep the original character
            formatted_text += char
    
    return formatted_text

def format_license(text):
    """
    DISABLED aggressive formatting to prevent 3->J errors
    This now only applies minimal, safe corrections
    """
    # if not text:
    #     return text
    
    # # Apply only very conservative corrections
    # return smart_format_license(text)
    return text

def read_license_plate(license_plate_crop):
    """
    Enhanced license plate text reading with multiple preprocessing attempts
    """
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None
    
    # Get multiple preprocessed versions
    processed_images = preprocess_license_plate(license_plate_crop)
    
    best_text = None
    best_score = 0
    best_method = None
    all_results = []
    
    # Try OCR on each preprocessed version
    for method_name, processed_img in processed_images:
        try:
            # Use EasyOCR with optimized settings for license plates
            detections = reader.readtext(
                processed_img,
                width_ths=0.4,      # Adjust width threshold
                height_ths=0.4,     # Adjust height threshold
                paragraph=False,     # Don't group text
                detail=1            # Return detailed results
            )
            
            for detection in detections:
                if len(detection) >= 3:
                    bbox, text, score = detection
                    
                    # Clean the text
                    cleaned_text = clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) >= 2:  # Minimum length
                        # Validate if it looks like a license plate
                        if validate_license_format(cleaned_text):
                            all_results.append((cleaned_text, score, method_name))
                            
                            print(f"    OCR ({method_name}): '{cleaned_text}' (raw: '{text}') score: {score:.3f}")
                            
                            if score > best_score:
                                best_text = cleaned_text
                                best_score = score
                                best_method = method_name
        
        except Exception as e:
            print(f"    OCR error with {method_name}: {e}")
            continue
    
    # If we found valid text, apply minimal formatting
    if best_text and best_score > 0.3:  # Lower threshold for debugging
        # Try to format the license plate (now very conservative)
        formatted_text = format_license(best_text)
        
        # Use formatted version if it still looks valid
        if license_complies_format(formatted_text):
            if formatted_text != best_text:
                print(f"    Formatted: '{best_text}' -> '{formatted_text}'")
            return formatted_text, best_score
        else:
            # Use original if formatting made it invalid
            return best_text, best_score
    
    # If no good results, return the best attempt even if low confidence
    if all_results:
        # Sort by score and return best
        all_results.sort(key=lambda x: x[1], reverse=True)
        best_attempt = all_results[0]
        print(f"    Best attempt: '{best_attempt[0]}' score: {best_attempt[1]:.3f}")
        return best_attempt[0], best_attempt[1]
    
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    
    foundIt = False
    best_match = None
    best_overlap = 0
    
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        
        # Check if license plate is inside vehicle bounding box
        if x1 >= xcar1 and y1 >= ycar1 and x2 <= xcar2 and y2 <= ycar2:
            # Calculate overlap area for better matching
            overlap_area = (min(x2, xcar2) - max(x1, xcar1)) * (min(y2, ycar2) - max(y1, ycar1))
            
            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_match = j
                foundIt = True
    
    if foundIt:
        return vehicle_track_ids[best_match]
    
    return -1, -1, -1, -1, -1