import numpy as np
from pydantic_model import FashionInput

CATEGORY_MAP = {
    'hair_colour': np.array(['Brown', 'Red', 'Blonde', 'Grey']),
    'eye_colour': np.array(['Green', 'Blue', 'Hazel', 'Grey', 'Black', 'Light Brown', 'Light Blue']),
    'skin_tone': np.array(['Fair', 'Medium', 'Olive', 'Brown', 'Very Dark']),
    'under_tone': np.array(['Cool', 'Neutral']),
    'torso_length': np.array(['Long Torso', 'Balanced'])
}

OUTPUT_MAP = {
    'Recommended Clothing Colors': np.array([
        'Earth Tones, Olive, Coral, Peach, Mustard, Warm Red',
        'Jewel Tones, Icy Blue, Lavender, Silver, Emerald',
        'Soft Pinks, Plums, Teal, Neutral Beige'
    ]),
    'Avoid Clothing Colors': np.array([
        'Cool Blue, Icy Gray, Jewel Tones',
        'Orange, Mustard, Brown',
        'Fluorescents, Harsh Yellow'
    ]),
    'Recommended Materials': np.array([
        'Structured Cotton',
        'Flowy Fabric',
        'Lightweight Cotton',
        'Jersey, Knits',
        'Soft Blends',
        'Stretchy, Soft Fabric',
        'Drapey Fabric'
    ]),
    'Recommended Patterns': np.array([
        'Curved Lines',
        'Vertical Stripes',
        'Bright Tops',
        'Diagonal Lines',
        'Simple Solids',
        'Subtle Prints',
        'Dark Solid Tops'
    ]),
    'Recommended Jewelry Metal': np.array([
        'Gold',
        'Silver',
        'Rose Gold'
    ]),
    'Recommended Clothing Color Wheel Region': np.array([
        'Warm colors (red, orange, yellow, warm greens)',
        'Cool colors (blue, green, violet, cool grays)',
        'Neutral-friendly zones (balanced warm/cool like teal, plum, taupe)'
    ])
}


def convert(v: FashionInput) -> np.ndarray:
    return np.concatenate([
        (cats == getattr(v, attr)).astype(int) 
        for attr, cats in CATEGORY_MAP.items()
    ])

def y_convert(model_output_dict: dict) -> dict:
    """
    Takes the dictionary directly from your ANN output, rounds/casts 
    the values, and maps them to their respective text categories.
    """
    
    decoded_output = {}
    
    for key, categories in OUTPUT_MAP.items():
        if key in model_output_dict:
            raw_array = model_output_dict[key]
            
            # 2. Round it and cast to integer 1s and 0s
            raw_array[raw_array == np.max(raw_array)] = 1

            binary_array = raw_array.round().astype(int)
            
            # 3. Squeeze out the batch dimension so np.where works seamlessly
            matched_indices = np.where(binary_array.squeeze() == 1)[0]
            
            # 4. Map to text strings
            decoded_output[key] = [categories[i] for i in matched_indices]
            
    return decoded_output

def get_y_dict(Y_matrix):
    return{
    'Recommended Clothing Colors':             Y_matrix[:, 0:3],   # 3 columns
    'Avoid Clothing Colors':                   Y_matrix[:, 3:6],   # 3 columns
    'Recommended Materials':                   Y_matrix[:, 6:13],  # 7 columns
    'Recommended Patterns':                    Y_matrix[:, 13:20], # 7 columns
    'Recommended Jewelry Metal':               Y_matrix[:, 20:23], # 3 columns
    'Recommended Clothing Color Wheel Region': Y_matrix[:, 23:26]  # 3 columns
}