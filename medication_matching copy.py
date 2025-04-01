from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np

fda_df = pd.read_csv('FDA_ALL.csv')
rx_df = pd.read_csv('RX_ALL.csv')
drugdata_df = pd.read_csv('drug_data.csv')


def match_with_fda(match_string, category=None, indexes=None):
    matches = []
    scores = []

    if indexes is None:
        indexes = list(range(len(fda_df)))

    for i in indexes:
        
        if category is None:
            medication_entry = str(fda_df.iloc[i]['Generic Name']) + str(fda_df.iloc[i]['Brand Name']) + str(fda_df.iloc[i]['Manufacturer']) + str(fda_df.iloc[i]['Dosage Strength'])  + str(fda_df.iloc[i]['Packaging'])
        else:
            medication_entry = str(fda_df.iloc[i][category])
        score = fuzz.token_set_ratio(medication_entry.lower(), match_string.lower())

        matches.append(fda_df.iloc[i]['INDEX'])
        scores.append(score)
    
    matches, scores = zip(*sorted(zip(matches, scores), key=lambda x: x[1], reverse=True))
    return matches, scores

from fuzzywuzzy import fuzz

def match_with_fda_hierarchical(match_string, threshold=70):
    match_string = match_string.lower().strip()
    
    matches = []
    scores = []

    def safe_str(value):
        """ Convert NaN or None to an empty string """
        return str(value).lower().strip() if pd.notna(value) else ""

    # Stage 1: Match by Generic Name + Brand Name (High Priority)
    first_pass = []
    for i in range(len(fda_df)):
        generic_brand = f"{safe_str(fda_df.iloc[i]['Generic Name'])} {safe_str(fda_df.iloc[i]['Brand Name'])}"
        score = fuzz.token_set_ratio(generic_brand, match_string)
        if score >= threshold:
            first_pass.append((i, score))

    if not first_pass:  # If no strong matches, relax threshold slightly
        first_pass = [(i, fuzz.token_set_ratio(safe_str(fda_df.iloc[i]['Generic Name']), match_string)) 
                      for i in range(len(fda_df))]

    # Stage 2: Filter by Pharmacologic Category
    second_pass = []
    for i, prev_score in first_pass:
        category = safe_str(fda_df.iloc[i]['Pharmacologic Category'])
        score = fuzz.token_set_ratio(category, match_string)
        final_score = (prev_score * 0.7) + (score * 0.3)
        if score >= threshold:
            second_pass.append((i, final_score))

    # Stage 3: Manufacturer & Dosage Strength
    third_pass = []
    for i, prev_score in second_pass:
        manufacturer = safe_str(fda_df.iloc[i]['Manufacturer'])
        dosage = safe_str(fda_df.iloc[i]['Dosage Strength'])

        manufacturer_score = fuzz.token_set_ratio(manufacturer, match_string)
        dosage_score = fuzz.token_set_ratio(dosage, match_string)

        final_score = (prev_score * 0.6) + (manufacturer_score * 0.15) + (dosage_score * 0.25)

        if dosage_score >= threshold or manufacturer_score >= threshold:
            third_pass.append((i, final_score))

    # Stage 4: Final Refinement with Packaging & Importer
    for i, prev_score in third_pass:
        packaging = safe_str(fda_df.iloc[i]['Packaging'])
        importer = safe_str(fda_df.iloc[i]['Importer'])

        packaging_score = fuzz.token_set_ratio(packaging, match_string)
        importer_score = fuzz.token_set_ratio(importer, match_string)

        final_score = (prev_score * 0.7) + (packaging_score * 0.2) + (importer_score * 0.1)

        matches.append(fda_df.iloc[i]['INDEX'])
        scores.append(round(final_score, 1))

    # Sort by final score
    if matches:
        matches, scores = zip(*sorted(zip(matches, scores), key=lambda x: x[1], reverse=True))
    
    return matches, scores


def match_with_rx(match_string):
    matches = []
    scores = []

    for i in range(len(drugdata_df)):
        
        medication_entry = str(drugdata_df.iloc[i]['Name'])
        score = fuzz.ratio(medication_entry.lower(), match_string.lower())
        #matches.append(rx_df.iloc[i]['INDEX'])
        matches.append(i)
        scores.append(score)
    
    matches, scores = zip(*sorted(zip(matches, scores), key=lambda x: x[1], reverse=True))
    return matches, scores

def get_info(match_string, limit=3):
    fda_matches, fda_scores = match_with_fda_hierarchical(match_string=match_string)
    matches = []

    for match, score in zip(fda_matches[:limit], fda_scores[:limit]):
        match_entry = dict(fda_df.iloc[match])
        match_entry['match_score'] = score
        generic_name = match_entry['Generic Name']
        rx_matches, _ = match_with_rx(generic_name)
        best_rx_match = dict(drugdata_df.iloc[rx_matches[0]])
        match_entry['rx_info'] = best_rx_match
        matches.append(match_entry)

    return {"matches": matches}

def get_info2(match_string, top_n=10, limit=3):
    search_terms = match_string.split('|')
    search_terms.insert(2, search_terms[1])
    search_categories = ['Generic Name', 'Manufacturer', 'Importer', 'Dosage Strength', 'Packaging']

    #filtered_scores = [0 for _ in range(len(fda_df))]
    filtered_matches = list(range(len(fda_df)))

    # for term, cat in zip(search_terms, search_categories):
    #     matches, scores = match_with_fda(term, cat, filtered_matches)
    #     filtered_matches, filtered_scores = matches[:top_n], scores[:top_n]
    #     cumm_scores = list(np.array(scores[:top_n]) + np.array(filtered_scores[:top_n]))
    #     filtered_matches, filtered_scores = matches[:top_n], cumm_scores[:top_n]
        
    # filtered_scores = list(np.array(filtered_scores)/len(search_categories))

    weights = np.array([0.3, 0.3, 0.1, 0.2, 0.1])  # Weights for each category
    filtered_scores = np.zeros(top_n)  # Initialize scores

    for i, (term, cat) in enumerate(zip(search_terms, search_categories)):
        matches, scores = match_with_fda(term, cat, filtered_matches)
        scores = np.array(scores[:top_n]) * weights[i]  # Apply weight
        filtered_scores += scores  # Accumulate weighted scores
        filtered_matches = matches[:top_n]  # Keep track of filtered matches

    filtered_scores = list(filtered_scores)  # Convert back to list if needed


    matches = []

    for match, score in zip(filtered_matches[:limit], filtered_scores[:limit]):
        match_entry = dict(fda_df.iloc[match])
        match_entry['match_score'] = score
        generic_name = match_entry['Generic Name']
        rx_matches, _ = match_with_rx(generic_name)
        best_rx_match = dict(drugdata_df.iloc[rx_matches[0]])
        match_entry['rx_info'] = best_rx_match
        matches.append(match_entry)

    return matches

#print(get_info('Allopurinol Llanole 100 mg Tablet Amherst Laboratories, Inc. Allopurinol'))