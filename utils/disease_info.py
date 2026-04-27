DISEASE_INFO = {
    "Athlete-foot": {
        "description": "Athlete foot (tinea pedis) is a fungal infection that usually begins between the toes and can spread to the soles and sides of the feet.",
        "symptoms": ["Itching", "Burning", "Cracked skin between toes", "Scaling", "Blisters"],
        "causes": ["Fungal infection (dermatophytes)", "Warm, moist footwear", "Walking barefoot in shared areas"],
        "remedies": [
            "Keep feet clean and dry",
            "Use antifungal creams or sprays",
            "Change socks daily",
            "Disinfect shoes and avoid tight footwear",
            "Seek medical advice if not improving"
        ],
        "severity": "Mild to Moderate",
        "color": "#34D399"
    },
    "Cellulitis": {
        "description": "Cellulitis is a bacterial skin infection that causes redness, swelling, warmth, and pain, often on the legs or arms.",
        "symptoms": ["Red, swollen skin", "Warmth", "Pain or tenderness", "Fever", "Blistering in severe cases"],
        "causes": ["Bacteria entering through cuts", "Insect bites", "Skin conditions that break the skin", "Weakened immunity"],
        "remedies": [
            "Seek medical care promptly",
            "Keep the affected area elevated",
            "Follow prescribed antibiotics",
            "Keep skin clean and covered",
            "Monitor for spreading redness or fever"
        ],
        "severity": "Moderate to Severe",
        "color": "#F97316"
    },
    "Chickenpox": {
        "description": "Chickenpox is a highly contagious disease caused by the varicella-zoster virus. It causes an itchy blister-like rash that first appears on the chest, back, and face.",
        "symptoms": ["Itchy blister-like rash", "Fever", "Tiredness", "Loss of appetite", "Headache"],
        "causes": ["Varicella-zoster virus", "Direct contact with rash", "Airborne transmission", "Contact with fluid from blisters"],
        "remedies": [
            "Apply calamine lotion to soothe itching",
            "Take oatmeal baths",
            "Keep fingernails short to prevent scratching",
            "Use antihistamines for itching relief",
            "Stay hydrated and get plenty of rest"
        ],
        "severity": "Mild to Moderate",
        "color": "#EAB308"
    },
    "Cutaneous-larva-migrans": {
        "description": "Cutaneous larva migrans is a skin infection caused by hookworm larvae that creates winding, itchy tracks under the skin.",
        "symptoms": ["Itchy, winding rash", "Raised red tracks", "Localized swelling", "Burning sensation"],
        "causes": ["Contact with contaminated soil or sand", "Walking barefoot on beaches", "Exposure to animal feces"],
        "remedies": [
            "See a clinician for antiparasitic treatment",
            "Avoid scratching to prevent infection",
            "Keep the area clean",
            "Wear footwear in sandy areas",
            "Treat pets for hookworms"
        ],
        "severity": "Moderate",
        "color": "#38BDF8"
    },
    "Impetigo": {
        "description": "Impetigo is a contagious bacterial skin infection that causes red sores which can burst and form honey-colored crusts.",
        "symptoms": ["Red sores", "Honey-colored crusts", "Itching", "Oozing", "Swollen lymph nodes"],
        "causes": ["Staph or strep bacteria", "Cuts or insect bites", "Close contact", "Poor hygiene"],
        "remedies": [
            "Keep sores clean and covered",
            "Use prescribed topical or oral antibiotics",
            "Wash hands frequently",
            "Avoid sharing towels and clothing",
            "Trim nails to reduce scratching"
        ],
        "severity": "Mild to Moderate",
        "color": "#FB7185"
    },
    "Nail-fungus": {
        "description": "Nail fungus (onychomycosis) is a fungal infection that causes nails to thicken, discolor, and become brittle.",
        "symptoms": ["Thickened nails", "Yellow or white discoloration", "Brittle nails", "Distorted nail shape", "Foul odor"],
        "causes": ["Fungal infection", "Moist environments", "Tight footwear", "Nail injury"],
        "remedies": [
            "Keep nails trimmed and dry",
            "Use antifungal treatments as prescribed",
            "Disinfect nail tools",
            "Wear breathable footwear",
            "Be consistent with treatment duration"
        ],
        "severity": "Mild to Moderate",
        "color": "#A78BFA"
    },
    "Ringworm": {
        "description": "Ringworm (tinea corporis) is a fungal infection of the skin. It's not caused by a worm. The infection causes a ring-shaped rash that is red and itchy.",
        "symptoms": ["Ring-shaped rash", "Itchy skin", "Red, scaly patches", "Hair loss in patches", "Brittle nails"],
        "causes": ["Fungal infection (dermatophytes)", "Contact with infected person or animal", "Sharing personal items", "Warm, humid environments"],
        "remedies": [
            "Apply antifungal creams",
            "Keep the affected area clean and dry",
            "Avoid sharing towels or clothing",
            "Wash hands frequently",
            "Complete the full course of antifungal treatment"
        ],
        "severity": "Mild",
        "color": "#22C55E"
    },
    "Shingles": {
        "description": "Shingles (herpes zoster) is a painful rash caused by reactivation of the varicella-zoster virus.",
        "symptoms": ["Pain or tingling", "Red rash", "Fluid-filled blisters", "Sensitivity to touch", "Fatigue"],
        "causes": ["Reactivation of varicella-zoster virus", "Weakened immunity", "Stress", "Older age"],
        "remedies": [
            "Seek medical care early",
            "Use prescribed antivirals",
            "Keep rash clean and dry",
            "Apply cool, wet compresses",
            "Avoid close contact with high-risk people"
        ],
        "severity": "Moderate to Severe",
        "color": "#F87171"
    }
}

def get_disease_names():
    return list(DISEASE_INFO.keys())

def get_disease_details(disease_name):
    return DISEASE_INFO.get(disease_name, None)
