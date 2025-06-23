import ocr_module
import medication_matching
from fuzzywuzzy import fuzz
import os

def get_info(image_path):

    ocr_text = ocr_module.extract_image_text(image_path=image_path)

    med_info = medication_matching.get_info(ocr_text, threshold=60)

    image_matches = []

    threshold = 60
    available_package_images_names = os.listdir('MedsForAll_Images')

    for match in med_info['matches']:
        match_name = f"{match['Generic Name']} {match['Brand Name']} {match['Dosage Strength']}"
        best_image_match = None
        best_match_score = 0
        for package_name in available_package_images_names:
            
            score = fuzz.token_sort_ratio(match_name, package_name[:-3].replace('-', '').strip())
            if score > best_match_score:
                best_match_score = score
                best_image_match = package_name

        if best_match_score >= threshold:
            image_matches.append(f'/package-images/{best_image_match}')
        else:
            image_matches.append('')

    med_info['images'] = image_matches

    return med_info

