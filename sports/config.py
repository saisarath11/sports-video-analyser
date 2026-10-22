"""
Sport-specific configurations and metrics definition.
Supports: Football, Cricket, Basketball, Badminton, Tennis.
"""

from typing import Dict, Any, List, Tuple

SPORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "football": {
        "display_name": "Football (Soccer)",
        "icon": "⚽",
        "description": "Pitch coverage, sprints, passing position analysis, work rate evaluation.",
        "detection_classes": [0],  # Person
        "detect_ball": True,
        "ball_class": 32,  # COCO sports ball
        "event_types": [
            "running", "sprinting", "jogging", "walking", "stationary", 
            "direction_change", "sprint_burst", "defensive_positioning"
        ],
        "speed_thresholds": {
            "stationary": 5.0,
            "walking": 30.0,
            "jogging": 80.0,
            "running": 130.0,
            "sprinting": 180.0
        },
        "performance_weights": {
            "movement_score": 0.30,
            "activity_score": 0.25,
            "coverage_score": 0.20,
            "consistency_score": 0.15,
            "posture_score": 0.10
        },
        "key_metrics": ["Distance Covered", "Sprint Count", "Top Speed", "Pitch Coverage Area", "Active Ratio"],
        "training_suggestions": {
            "high_movement": [
                "Maintain high endurance levels with high-intensity interval training (HIIT).",
                "Focus on tactical positioning to optimize energy output."
            ],
            "low_movement": [
                "Incorporate sprint interval drills to build explosive speed.",
                "Work on off-the-ball movement to stay involved in active play phases."
            ],
            "good_coverage": [
                "Great spatial awareness. Continue supporting both offensive and defensive transitions."
            ],
            "poor_coverage": [
                "Expand work area by overlapping into wide channels during attacks."
            ],
            "consistent": [
                "Strong work rate throughout the match section."
            ],
            "inconsistent": [
                "Focus on stamina conditioning to avoid fatigue in final segments."
            ]
        },
        "strengths_templates": {
            "high_movement": ["High movement activity across pitch", "Strong distance coverage"],
            "high_speed": ["Explosive sprint capability", "Fast transition pace"]
        },
        "improvement_templates": {
            "low_movement": ["Low total distance covered", "Periods of static positioning"],
            "low_speed": ["Limited high-speed bursts", "Slower recovery pace"]
        }
    },
    "cricket": {
        "display_name": "Cricket",
        "icon": "🏏",
        "description": "Batting posture, bowling action analysis, fielding movement, running between wickets.",
        "detection_classes": [0],
        "detect_ball": True,
        "ball_class": 32,
        "event_types": [
            "batting_stance", "bowling_action", "fielding_movement", 
            "running_between_wickets", "stationary", "ready_position", "direction_change"
        ],
        "speed_thresholds": {
            "stationary": 5.0,
            "walking": 25.0,
            "jogging": 70.0,
            "running": 120.0,
            "sprinting": 170.0
        },
        "performance_weights": {
            "movement_score": 0.20,
            "activity_score": 0.20,
            "coverage_score": 0.15,
            "consistency_score": 0.25,
            "posture_score": 0.20
        },
        "key_metrics": ["Sprint Acceleration", "Stance Balance", "Fielding Radius", "Reaction Speed"],
        "training_suggestions": {
            "high_movement": [
                "Excellent energy between wickets and in fielding.",
                "Maintain core stability during quick turns."
            ],
            "low_movement": [
                "Work on quick acceleration off the crease.",
                "Improve reaction time in close-in fielding positions."
            ],
            "good_coverage": ["Wide fielding coverage; strong lateral movement."],
            "poor_coverage": ["Practice lateral agility drills for improved boundary coverage."],
            "consistent": ["Rhythmic run-up and balanced landing stance."],
            "inconsistent": ["Work on run-up consistency and post-shot recovery speed."]
        },
        "strengths_templates": {
            "high_movement": ["Fast running between wickets", "Agile boundary fielding"],
            "high_speed": ["Explosive quick-single acceleration", "Rapid chase pace"]
        },
        "improvement_templates": {
            "low_movement": ["Hesitant running between wickets", "Static fielding stance"],
            "low_speed": ["Delayed initial burst", "Slow turn at the crease"]
        }
    },
    "basketball": {
        "display_name": "Basketball",
        "icon": "🏀",
        "description": "Court coverage, quick direction changes, defensive stance, transition speed.",
        "detection_classes": [0],
        "detect_ball": True,
        "ball_class": 32,
        "event_types": [
            "running", "sprinting", "defensive_stance", "jump_attempt", 
            "direction_change", "dribbling_movement", "stationary"
        ],
        "speed_thresholds": {
            "stationary": 6.0,
            "walking": 35.0,
            "jogging": 90.0,
            "running": 140.0,
            "sprinting": 190.0
        },
        "performance_weights": {
            "movement_score": 0.25,
            "activity_score": 0.30,
            "coverage_score": 0.15,
            "consistency_score": 0.15,
            "posture_score": 0.15
        },
        "key_metrics": ["Court Transition Time", "Lateral Shuffle Speed", "Active Work Ratio", "Vertical Explosiveness"],
        "training_suggestions": {
            "high_movement": [
                "Great high-tempo pace in fast breaks.",
                "Keep working on controlled deceleration to prevent fouls."
            ],
            "low_movement": [
                "Increase off-ball movement to open up passing lanes.",
                "Work on defensive slide speed across the key."
            ],
            "good_coverage": ["Full court engagement on both ends of the floor."],
            "poor_coverage": ["Focus on sprinting back on transition defense."],
            "consistent": ["High intensity maintained through multi-possession stretches."],
            "inconsistent": ["Build cardio conditioning for sustained high-press defense."]
        },
        "strengths_templates": {
            "high_movement": ["Rapid fast-break pace", "Constant offensive cutting"],
            "high_speed": ["Sharp directional cuts", "Quick floor transition"]
        },
        "improvement_templates": {
            "low_movement": ["Slow transition drop-back", "Static standing on perimeter"],
            "low_speed": ["Slow lateral shuffle", "Delayed defensive recovery"]
        }
    },
    "badminton": {
        "display_name": "Badminton",
        "icon": "🏸",
        "description": "Court coverage, footwork agility, split-step readiness, net & baseline movements.",
        "detection_classes": [0],
        "detect_ball": False,
        "ball_class": None,
        "event_types": [
            "ready_position", "lateral_movement", "forward_movement", 
            "backward_movement", "jump_smash_prep", "net_approach", 
            "baseline_return", "direction_change", "stationary"
        ],
        "speed_thresholds": {
            "stationary": 4.0,
            "walking": 20.0,
            "jogging": 60.0,
            "running": 110.0,
            "sprinting": 160.0
        },
        "performance_weights": {
            "movement_score": 0.20,
            "activity_score": 0.25,
            "coverage_score": 0.25,
            "consistency_score": 0.15,
            "posture_score": 0.15
        },
        "key_metrics": ["Center Recovery Speed", "Corner Reach Efficiency", "Split-Step Frequency", "Footwork Agility"],
        "training_suggestions": {
            "high_movement": [
                "Sharp court coverage and quick footwork recovery to center.",
                "Maintain low center of gravity during net lunges."
            ],
            "low_movement": [
                "Focus on active split-step before opponent contact.",
                "Practice shadow footwork for 6-corner court coverage."
            ],
            "good_coverage": ["Excellent court coverage across all 4 corners."],
            "poor_coverage": ["Improve rear-court footwork transitions to avoid late clears."],
            "consistent": ["Balanced stance maintained throughout rallies."],
            "inconsistent": ["Work on posture reset after lunges."]
        },
        "strengths_templates": {
            "high_movement": ["Rapid center recovery", "Agile net lunges"],
            "high_speed": ["Explosive push-offs", "Quick corner transitions"]
        },
        "improvement_templates": {
            "low_movement": ["Late recovery to base", "Heavy footwork movement"],
            "low_speed": ["Slow rear-court backpedal", "Delayed lunge recovery"]
        }
    },
    "tennis": {
        "display_name": "Tennis",
        "icon": "🎾",
        "description": "Baseline movement, split-step timing, net approach, lateral court movement.",
        "detection_classes": [0],
        "detect_ball": True,
        "ball_class": 32,
        "event_types": [
            "ready_position", "baseline_movement", "net_approach", 
            "lateral_shuffle", "direction_change", "serve_preparation", "stationary"
        ],
        "speed_thresholds": {
            "stationary": 4.5,
            "walking": 22.0,
            "jogging": 65.0,
            "running": 115.0,
            "sprinting": 165.0
        },
        "performance_weights": {
            "movement_score": 0.25,
            "activity_score": 0.25,
            "coverage_score": 0.20,
            "consistency_score": 0.15,
            "posture_score": 0.15
        },
        "key_metrics": ["Baseline Rally Speed", "Lateral Shuffle Agility", "Net Approach Rate", "Recovery to Center"],
        "training_suggestions": {
            "high_movement": [
                "Great footwork mobility along the baseline.",
                "Maintain weight transfer on wide forehand recovery."
            ],
            "low_movement": [
                "Incorporate lateral crossover steps for wide ball reach.",
                "Focus on aggressive net approach movement."
            ],
            "good_coverage": ["Strong baseline coverage and forward net transition."],
            "poor_coverage": ["Work on recovery footwork to the center hash mark."],
            "consistent": ["Disciplined ready posture between rally shots."],
            "inconsistent": ["Avoid flat-footed stance during long exchanges."]
        },
        "strengths_templates": {
            "high_movement": ["Relentless baseline coverage", "Swift net approach"],
            "high_speed": ["Sharp lateral crossover steps", "Fast rally recovery"]
        },
        "improvement_templates": {
            "low_movement": ["Static baseline stance", "Slow recovery to center"],
            "low_speed": ["Sluggish wide-ball pursuit", "Delayed split-step execution"]
        }
    }
}


def get_sport_config(sport: str) -> Dict[str, Any]:
    """Return configuration dictionary for specified sport."""
    return SPORT_CONFIG.get(sport.lower(), SPORT_CONFIG["football"])


def list_sports() -> List[Tuple[str, str, str]]:
    """Return list of (key, display_name, icon) tuples."""
    return [(k, v["display_name"], v["icon"]) for k, v in SPORT_CONFIG.items()]


def get_speed_category(speed_px_s: float, sport: str) -> str:
    """Classify movement speed using sport-specific thresholds."""
    config = get_sport_config(sport)
    thresholds = config["speed_thresholds"]
    
    if speed_px_s < thresholds["stationary"]:
        return "stationary"
    elif speed_px_s < thresholds["walking"]:
        return "walking"
    elif speed_px_s < thresholds["jogging"]:
        return "jogging"
    elif speed_px_s < thresholds["running"]:
        return "running"
    else:
        return "sprinting"
