"""
Unit tests for Performance Evaluation and Feedback Generator.
"""

from core.models import PerformanceIndicators, MotionMetrics
from core.feedback_generator import FeedbackGenerator
from utils.config import get_grade


def test_grade_thresholds():
    assert get_grade(95.0) == "A+"
    assert get_grade(85.0) == "A"
    assert get_grade(70.0) == "B"
    assert get_grade(55.0) == "C"
    assert get_grade(40.0) == "D"
    assert get_grade(20.0) == "E"


def test_template_feedback_generation():
    gen = FeedbackGenerator(sport="football", use_gemini=False)
    perf = PerformanceIndicators(player_id=1, overall_score=80.0, grade="A")
    motion = MotionMetrics(player_id=1, active_duration_s=20.0, total_duration_s=30.0, est_distance_m=150.0)
    
    fb = gen.generate_feedback(player_id=1, performance=perf, motion=motion, events=[])
    assert fb.player_id == 1
    assert fb.generated_by == "template"
    assert len(fb.summary) > 0
