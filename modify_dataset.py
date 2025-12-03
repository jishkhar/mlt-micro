import pandas as pd
import numpy as np

def modify_dataset(filepath='DWLR_Dataset_2023.csv'):
    """
    Adds a 'Location' attribute with random Indian states and adjusts 'Water_Level_m'
    to create a correlation between location and water level.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return

    # List of Indian states
    states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]

    # Define realistic base depth ranges (mbgl) for states
    # Low value = Shallow water (Good/Flooding)
    # High value = Deep water (Critical/Dry)
    state_base_depths = {
        'Andhra Pradesh': (5, 15),
        'Arunachal Pradesh': (2, 8),
        'Assam': (1, 4), # Very shallow, prone to flooding
        'Bihar': (3, 8),
        'Chhattisgarh': (4, 10),
        'Goa': (2, 8),
        'Gujarat': (15, 30), # Semi-arid
        'Haryana': (20, 40), # Critical
        'Himachal Pradesh': (5, 15),
        'Jharkhand': (5, 12),
        'Karnataka': (10, 25),
        'Kerala': (2, 8),
        'Madhya Pradesh': (8, 20),
        'Maharashtra': (10, 25),
        'Manipur': (2, 8),
        'Meghalaya': (1, 5),
        'Mizoram': (2, 8),
        'Nagaland': (2, 8),
        'Odisha': (3, 10),
        'Punjab': (20, 45), # Critical
        'Rajasthan': (30, 60), # Desert/Critical
        'Sikkim': (2, 8),
        'Tamil Nadu': (10, 25), # Critical in parts
        'Telangana': (10, 25),
        'Tripura': (2, 6),
        'Uttar Pradesh': (5, 15),
        'Uttarakhand': (5, 15),
        'West Bengal': (2, 8),
        'Delhi': (20, 40),
        'Chandigarh': (15, 30),
        'Puducherry': (5, 10)
    }

    print("Generating realistic data based on State and Season...")

    def get_water_level(row):
        state = row['Location']
        date = pd.to_datetime(row['Date'])
        month = date.month
        
        # Enforce SAFE levels for Monsoon and Post-Monsoon for ALL states
        if month in [6, 7, 8, 9]: # Monsoon
            # Safe range: 0.5m to 3.5m (Very safe/flooding potential)
            depth = np.random.uniform(0.5, 3.5)
        elif month in [10, 11, 12]: # Post-Monsoon
            # Safe range: 1.5m to 4.8m (Still safe, slightly deeper)
            depth = np.random.uniform(1.5, 4.8)
        else:
            # Winter/Pre-Monsoon: Use state-specific base depths (can be critical)
            min_d, max_d = state_base_depths.get(state, (5, 15))
            base_depth = np.random.uniform(min_d, max_d)
            
            if month in [1, 2]: # Winter
                season_factor = 1.0
            else: # Mar, Apr, May (Pre-Monsoon/Summer)
                season_factor = 1.3 # Deeper
                
            depth = base_depth * season_factor
        
        # Add noise
        depth += np.random.normal(0, 0.3)
        
        return max(0.1, depth) # Ensure positive depth

    df['Water_Level_m'] = df.apply(get_water_level, axis=1)

    # Save the modified dataset
    df.to_csv(filepath, index=False)
    print(f"\nSuccessfully modified {filepath} with 'Location' attribute and adjusted water levels.")
    print(df.head())

if __name__ == "__main__":
    modify_dataset()
