"""
Personalized AI Feedback Generator using Google Gemini API with fallback to template engine.
"""

import json
import re
from typing import Dict, List
from core.models import PerformanceIndicators, MotionMetrics, PlayerFeedback, AnalysisResult
from sports.config import get_sport_config
from utils.config import GEMINI_API_KEY, GEMINI_MODEL
from utils.logging_utils import logger

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class FeedbackGenerator:
    """
    Generates personalized recommendations and coaching feedback.
    Uses Gemini API when configured, otherwise uses template engine.
    """
    def __init__(self, sport: str = "football", use_gemini: bool = True):
        self.sport = sport
        self.config = get_sport_config(sport)
        self.use_gemini = use_gemini and GEMINI_AVAILABLE and bool(GEMINI_API_KEY)
        self.gemini_model = None

        if self.use_gemini:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
                logger.info(f"Initialized Gemini AI feedback engine ({GEMINI_MODEL})")
            except Exception as e:
                logger.warning(f"Gemini API initialization failed: {e}. Falling back to templates.")
                self.use_gemini = False

    def generate_feedback(self, player_id: int, performance: PerformanceIndicators,
                          motion: MotionMetrics, events: list) -> PlayerFeedback:
        """Generate feedback for a single player."""
        if self.use_gemini and self.gemini_model:
            try:
                return self._generate_gemini_feedback(player_id, performance, motion, events)
            except Exception as e:
                logger.warning(f"Gemini generation error: {e}. Falling back to template.")

        return self._generate_template_feedback(player_id, performance, motion, events)

    def generate_all_feedback(self, result: AnalysisResult) -> Dict[int, PlayerFeedback]:
        """Generate feedback for all tracked players in AnalysisResult."""
        feedback_dict = {}
        for pid in result.get_player_ids():
            perf = result.performance.get(pid)
            motion = result.motion_metrics.get(pid)
            events = result.get_events_for_player(pid)
            if perf and motion:
                feedback_dict[pid] = self.generate_feedback(pid, perf, motion, events)
        return feedback_dict

    def _generate_template_feedback(self, player_id: int, performance: PerformanceIndicators,
                                     motion: MotionMetrics, events: list) -> PlayerFeedback:
        """Generate template-driven feedback."""
        sport_name = self.config["display_name"]
        suggestions = self.config["training_suggestions"]

        summary = (
            f"Player {player_id} completed the {sport_name} session with an AI prototype score of "
            f"{performance.overall_score}/100 (Grade: {performance.grade}). "
            f"Active movement duration was recorded at {motion.active_duration_s} seconds with an estimated "
            f"total distance of {motion.est_distance_m} metres."
        )

        train_suggs = suggestions.get("high_movement", ["Incorporate dynamic agility drills."]) if motion.active_duration_s > 10 else suggestions.get("low_movement", ["Focus on acceleration drills."])
        
        recs = [
            f"Consistently track video metrics to monitor {sport_name} improvement.",
            "Review movement trajectory heatmaps with your coach."
        ]

        return PlayerFeedback(
            player_id=player_id,
            sport=self.sport,
            summary=summary,
            strengths=performance.strengths,
            improvements=performance.weaknesses,
            training_suggestions=train_suggs,
            recommendations=recs,
            generated_by="template"
        )

    def _generate_gemini_feedback(self, player_id: int, performance: PerformanceIndicators,
                                   motion: MotionMetrics, events: list) -> PlayerFeedback:
        """Generate feedback using Google Gemini LLM API."""
        prompt = f"""
You are an AI Sports Performance Coach analyzing a video of a {self.config['display_name']} player.
Extracted Video Analysis Data for Player {player_id}:
- Prototype Performance Score: {performance.overall_score}/100 (Grade: {performance.grade})
- Movement Score: {performance.movement_score}/100
- Activity Score: {performance.activity_score}/100
- Coverage Score: {performance.coverage_score}/100
- Estimated Distance: {motion.est_distance_m} metres
- Estimated Peak Speed: {motion.est_max_speed_kmh} km/h
- Active Duration: {motion.active_duration_s} seconds
- Stationary Duration: {motion.stationary_duration_s} seconds
- Detected Strengths: {', '.join(performance.strengths)}
- Areas for Improvement: {', '.join(performance.weaknesses)}

Instructions:
1. Provide personalized coaching feedback.
2. Note that metrics are AI-generated prototype indicators.
3. Do NOT provide medical advice or injury diagnosis.
4. Output MUST be valid JSON only with keys: "summary", "strengths", "improvements", "training_suggestions", "recommendations".
"""

        response = self.gemini_model.generate_content(prompt)
        text = response.text.strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(text)

        return PlayerFeedback(
            player_id=player_id,
            sport=self.sport,
            summary=data.get("summary", ""),
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            training_suggestions=data.get("training_suggestions", []),
            recommendations=data.get("recommendations", []),
            generated_by="gemini"
        )
