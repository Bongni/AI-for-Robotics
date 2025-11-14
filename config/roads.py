road_elements_list_snake = [
    {"type":"curved", "curvature":-1/120.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":-1/100.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/80.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/60.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":-1/50.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":-1/40.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/30.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
]

road_elements_list_race = [
    {"type":"curved", "curvature":1/50.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"straight", "length":200.0, "v_max_kph":180.0},
    {"type":"curved", "curvature":1/60.0, "angle_in_degrees":90.0, "v_max_kph":80.0},
    {"type":"curved", "curvature":-1/60.0, "angle_in_degrees":90.0, "v_max_kph":80.0},
    {"type":"straight", "length":100.0, "v_max_kph":120.0},
    {"type":"curved", "curvature":1/60.0, "angle_in_degrees":90.0, "v_max_kph":80.0},
    {"type":"curved", "curvature":1/60.0, "angle_in_degrees":90.0, "v_max_kph":80.0},
    {"type":"straight", "length":100.0, "v_max_kph":120.0},
    {"type":"curved", "curvature":1/80.0, "angle_in_degrees":90.0, "v_max_kph":90.0},
    {"type":"curved", "curvature":-1/80.0, "angle_in_degrees":90.0, "v_max_kph":90.0},
    {"type":"curved", "curvature":-1/80.0, "angle_in_degrees":90.0, "v_max_kph":90.0},
    {"type":"curved", "curvature":1/80.0, "angle_in_degrees":90.0, "v_max_kph":90.0},
    {"type":"straight", "length":100.0, "v_max_kph":120.0},
]

road_elements_list_spiral = [
    {"type":"curved", "curvature":1/120.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/100.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/80.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/60.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/50.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/40.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
    {"type":"curved", "curvature":1/30.0, "angle_in_degrees":90.0, "v_max_kph":70.0},
]

road_elements_lists = [
    road_elements_list_snake,
    road_elements_list_race,
    road_elements_list_spiral,
]

road_elements_names = [
    "Snake",
    "Race",
    "Spiral",
]

assert len(road_elements_lists) == len(road_elements_names), "Number of road elements lists must match the number of names."

# Example 1: Standard Randomization
road_randomization_params_std = {
    'num_elements_range': (3, 6),          # Road will have 3 to 6 elements
    'straight_length_range': (100.0, 150.0), # Length of straight segments between 100 and 150 meters
    'curvature_range': (-1/300.0, 1/300.0), # Curvature between -1/300 and 1/300 for curves
    'angle_range': (20.0, 45.0)            # Angle of the curves between 20 and 45 degrees
}

# # Example 2: More Straight Roads
road_randomization_params_straight = {
    'num_elements_range': (2, 4),            # Fewer elements, 2 to 4
    'straight_length_range': (150.0, 300.0), # Longer straight segments between 150 and 300 meters
    'curvature_range': (-1/1000.0, 1/1000.0), # Small curvature for gentler curves
    'angle_range': (10.0, 30.0)              # Curves have lower angles between 10 and 30 degrees
}

# # Example 3: More challenging roads with Tighter curves
road_randomization_params_tight = {
    'num_elements_range': (4, 8),             # More road elements
    'straight_length_range': (50.0, 100.0),   # Shorter straight segments
    'curvature_range': (-1/100.0, 1/100.0),   # Tighter curves
    'angle_range': (30.0, 70.0)               # More aggressive curves with wider angle range
}

# Example 4: Completely Randomized
road_randomization_params_full = {
    'num_elements_range': (1, 10),             # Highly variable number of road elements
    'straight_length_range': (50.0, 300.0),    # Straight segments can be very short or long
    'curvature_range': (-1/50.0, 1/50.0),      # Very tight curves possible
    'angle_range': (10.0, 90.0)                # Wide range of angles for curves
}

road_elements_list_difficult = [
    {"type": "straight", "length": 50.0, 'v_max_kph': 60.0},
    {"type": "curved", "curvature": 1/300.0, "angle_in_degrees": 90.0, 'v_max_kph': 40.0},
    {"type": "straight", "length": 50.0, 'v_max_kph': 30.0, 'v_max_kph': 50.0},
    {"type": "curved", "curvature": -1/200.0, "angle_in_degrees": 120.0, 'v_max_kph': 40.0},
    {"type": "straight", "length": 50.0, 'v_max_kph': 60.0},
]