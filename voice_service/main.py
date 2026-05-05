from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from upstash_redis import Redis
import logging
import shap
import opensmile
from pydub import AudioSegment
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================
models = {}

_HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = str(_HERE / "models" / "audio_calibration_data_v1.json")


# ════════════════════════════════════════════════════════════════════════════════
# AUDIO FEEDBACK SYSTEM
# Group-based SHAP + golden zone feedback with human-friendly names and tips.
# ════════════════════════════════════════════════════════════════════════════════

FRIENDLY_NAMES = {
    # Loudness & Energy
    "loudness_sma3_percentile20.0":      "Quietest Moments",
    "loudness_sma3_percentile50.0":      "Typical Speaking Volume",
    "loudness_sma3_percentile80.0":      "Peak Speaking Volume",
    "loudness_sma3_stddevNorm":          "Volume Variety",
    "loudness_sma3_pctlrange0-2":        "Quiet-to-Loud Range",
    "loudness_sma3_meanRisingSlope":     "Sentence Energy Build-up",
    "loudness_sma3_stddevRisingSlope":   "Energy Build-up Consistency",
    "loudness_sma3_meanFallingSlope":    "Sentence Endings",
    "loudness_sma3_stddevFallingSlope":  "Sentence Ending Consistency",
    "loudnessPeaksPerSec":               "Emphasis & Stress",
    "loudness_dynamics_power":           "Overall Volume Dynamics",
    "vocal_projection":                  "How Well Your Voice Carries",
    # Pitch & Expressiveness
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "Pitch Range",
    "spectralFlux_sma3_amean":           "Tonal Variety",
    "spectralFlux_sma3_stddevNorm":      "Tonal Consistency",
    "alphaRatioV_sma3nz_amean":          "Voice Brightness",
    "alphaRatioV_sma3nz_stddevNorm":     "Voice Brightness Consistency",
    "hammarbergIndexV_sma3nz_stddevNorm":"Vocal Texture Consistency",
    "hammarbergIndexUV_sma3nz_amean":    "Pauses Between Words",
    "brightness_contrast":               "Voice Sharpness & Clarity",
    "slopeV0-500_sma3nz_stddevNorm":     "Bass Tone Variation",
    "slopeUV0-500_sma3nz_amean":         "Breath Sound Quality",
    "slopeUV500-1500_sma3nz_amean":      "Pause Sound Quality",
    # Voice Quality
    "HNRdBACF_sma3nz_amean":            "Voice Smoothness",
    "HNRdBACF_sma3nz_stddevNorm":       "Voice Smoothness Consistency",
    "shimmerLocaldB_sma3nz_amean":       "Voice Steadiness",
    "shimmerLocaldB_sma3nz_stddevNorm":  "Voice Steadiness Consistency",
    "vocal_instability":                 "Overall Vocal Stability",
    "F2amplitudeLogRelF0_sma3nz_amean":      "Vowel Strength",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": "Vowel Strength Variation",
    "F2bandwidth_sma3nz_stddevNorm":          "Vowel Resonance Consistency",
    "F2frequency_sma3nz_stddevNorm":          "Vowel Tone Variation",
    "F3amplitudeLogRelF0_sma3nz_amean":      "Voice Richness",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": "Voice Richness Variation",
    "F3bandwidth_sma3nz_stddevNorm":          "Vocal Depth Consistency",
    "F3frequency_sma3nz_stddevNorm":          "Vocal Depth Variation",
    "logRelF0-H1-H2_sma3nz_stddevNorm":      "Vocal Tone Balance",
    "mfcc1_sma3_stddevNorm":   "Voice Texture Variety",
    "mfcc1V_sma3nz_stddevNorm":"Voiced Tone Texture",
    "mfcc2V_sma3nz_stddevNorm":"Vocal Warmth Variation",
    "mfcc3_sma3_stddevNorm":   "Tonal Color Variety",
    "mfcc4V_sma3nz_stddevNorm":"Voice Character Variation",
    "mfcc4_sma3_stddevNorm":   "Voice Color Consistency",
    # Fluency & Flow
    "voiced_flow":                       "Speech Flow",
    "MeanUnvoicedSegmentLength":         "Pause Length",
}

FEEDBACK_GROUPS = {
    "Loudness & Energy": {
        "members": [
            "loudness_sma3_percentile20.0", "loudness_sma3_percentile50.0",
            "loudness_sma3_percentile80.0", "loudness_sma3_stddevNorm",
            "loudness_sma3_pctlrange0-2",   "loudness_sma3_meanRisingSlope",
            "loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope",
            "loudness_sma3_stddevFallingSlope","loudnessPeaksPerSec",
            "loudness_dynamics_power",       "vocal_projection",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0800,
    },
    "Pitch & Expressiveness": {
        "members": [
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "spectralFlux_sma3_amean",       "spectralFlux_sma3_stddevNorm",
            "brightness_contrast",
            "alphaRatioV_sma3nz_amean",      "alphaRatioV_sma3nz_stddevNorm",
            "hammarbergIndexV_sma3nz_stddevNorm", "hammarbergIndexUV_sma3nz_amean",
            "slopeV0-500_sma3nz_stddevNorm",
            "slopeUV0-500_sma3nz_amean",     "slopeUV500-1500_sma3nz_amean",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0600,
    },
    "Voice Quality": {
        "members": [
            "HNRdBACF_sma3nz_amean",      "HNRdBACF_sma3nz_stddevNorm",
            "shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma3nz_stddevNorm",
            "vocal_instability",
            "F2amplitudeLogRelF0_sma3nz_amean",      "F2amplitudeLogRelF0_sma3nz_stddevNorm",
            "F2bandwidth_sma3nz_stddevNorm",          "F2frequency_sma3nz_stddevNorm",
            "F3amplitudeLogRelF0_sma3nz_amean",      "F3amplitudeLogRelF0_sma3nz_stddevNorm",
            "F3bandwidth_sma3nz_stddevNorm",          "F3frequency_sma3nz_stddevNorm",
            "logRelF0-H1-H2_sma3nz_stddevNorm",
            "mfcc1_sma3_stddevNorm",     "mfcc1V_sma3nz_stddevNorm",
            "mfcc2V_sma3nz_stddevNorm",  "mfcc3_sma3_stddevNorm",
            "mfcc4V_sma3nz_stddevNorm",  "mfcc4_sma3_stddevNorm",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0700,
    },
    "Fluency & Flow": {
        "members": [
            "voiced_flow",
            "MeanUnvoicedSegmentLength",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0400,
    },
}

SKIP_COACHING = {
    "F2frequency_sma3nz_stddevNorm",
    "F3frequency_sma3nz_stddevNorm",
    "F2bandwidth_sma3nz_stddevNorm",
    "F3bandwidth_sma3nz_stddevNorm",
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "mfcc4_sma3_stddevNorm",
    "mfcc4V_sma3nz_stddevNorm",
}


# ── Coaching tip generator ────────────────────────────────────────────────────
def _get_coaching_tip(feat: str, val: float, zone: dict, direction: str) -> tuple[str, str]:
    """
    Returns (tip_text, status_colour).
    direction: "positive" = SHAP helped score, "negative" = SHAP hurt score.
    status_colour: "green" | "yellow" | "red"

    zone_direction comes directly from the calibration JSON — it is ground truth
    and is NOT overridden here. The "More is better / Less is better / Middle is best"
    label on the frontend reads zone_direction from the tip payload and should
    display whatever the JSON says.

    The nudge logic in positive SHAP tips uses zone_direction to tell the user
    which way to push — but only in the tip text, not by changing zone_direction.
    """
    friendly = FRIENDLY_NAMES.get(feat, feat)
    zone_min = zone.get("min", -99)
    zone_max = zone.get("max",  99)
    feat_dir = zone.get("direction", "INCREASING")  # ground truth from JSON — do not override
    in_zone  = zone_min <= val <= zone_max

    # ── Positive SHAP ─────────────────────────────────────────────────────────
    if direction == "positive":

        def _elite_nudge() -> str | None:
            """
            When SHAP is positive (feature is helping) but the value is outside
            the elite zone, tell the user they're doing well but can fine-tune
            toward the zone. The nudge direction is derived from where val sits
            relative to the zone — NOT from feat_dir — so it always makes sense
            regardless of what the JSON direction says.
            """
            if in_zone:
                return None  # already in zone, no nudge needed
            if val < zone_min:
                # Dot is left of zone — needs to go higher/more
                return (
                    "You're doing well here — pushing this a little higher "
                    "will bring you right into the elite range."
                )
            if val > zone_max:
                # Dot is right of zone — needs to come down/less
                return (
                    "You're doing well here — toning this down slightly "
                    "will land you right in the elite range."
                )
            return None

        nudge = _elite_nudge()

        # Feature-specific base praise tips
        base_tip = None
        if "loudnessPeaksPerSec" in feat:
            base_tip = "You naturally stress key words — this makes you sound engaging and easy to follow."
        elif "vocal_projection" in feat:
            base_tip = "Your voice carries well — you sound confident and easy to hear."
        elif "loudness_sma3_stddevNorm" in feat or "loudness_dynamics_power" in feat:
            base_tip = "You vary your volume nicely — your speech doesn't feel flat or robotic."
        elif "F0semitoneFrom27.5Hz_sma3nz_stddevNorm" in feat:
            base_tip = "Your pitch rises and falls naturally — you sound expressive and engaged."
        elif "spectralFlux_sma3_amean" in feat:
            base_tip = "Your voice has good tonal variety — listeners stay engaged throughout."
        elif "HNRdBACF_sma3nz_amean" in feat:
            base_tip = "Your voice sounds clean and smooth — there's no breathiness or roughness getting in the way."
        elif "shimmerLocaldB_sma3nz_amean" in feat or "shimmerLocaldB_sma3nz_stddevNorm" in feat:
            base_tip = "Your voice is steady and controlled — you sound composed and in command."
        elif "vocal_instability" in feat:
            base_tip = "Your voice stays rock-steady — no shakiness or wobbling getting in the way."
        elif "voiced_flow" in feat:
            base_tip = "Your speech flows smoothly — sentences connect naturally without choppy breaks."
        elif "MeanUnvoicedSegmentLength" in feat:
            base_tip = "Your pauses are well-timed — just long enough to land your points without losing momentum."
        elif "F2amplitudeLogRelF0_sma3nz_amean" in feat or "F3amplitudeLogRelF0_sma3nz_amean" in feat:
            base_tip = "Your vowels sound full and resonant — this gives your voice depth and presence."
        elif "mfcc" in feat.lower():
            base_tip = "Your voice has a pleasant, consistent texture — it's easy and comfortable to listen to."
        elif "brightness_contrast" in feat or "alphaRatioV_sma3nz_amean" in feat:
            base_tip = "Your voice has a clear, forward quality — you sound articulate and alert."

        if base_tip is None:
            base_tip = (
                f"Your {friendly} is working in your favour — "
                "it's well-calibrated for a confident delivery."
            )

        if nudge:
            return f"{base_tip} {nudge}", "yellow"
        return base_tip, "green"

    # ── Negative SHAP ─────────────────────────────────────────────────────────
    # If value is already in zone, don't tell them to change it.
    if in_zone:
        return (
            f"Your {friendly} is actually within the ideal range — "
            "this might be slightly affecting your score due to other factors. "
            "Keep doing what you're doing here.", "yellow"
        )

    # ── LOUDNESS & ENERGY ─────────────────────────────────────────────────────
    if "loudnessPeaksPerSec" in feat:
        if val > zone_max:
            return (
                "You're emphasising too many words — when everything is stressed, nothing stands out. "
                "Choose your 2–3 most important words per sentence and stress only those.", "yellow"
            )
        return (
            "You're not stressing words enough — everything lands with equal weight, "
            "which can sound monotone. Try this: pick the 2–3 most important words in each "
            "sentence and say them noticeably louder. Practice on one sentence at a time.", "red"
        )

    if "vocal_projection" in feat:
        if val > zone_max:
            return (
                "Your voice is projecting very strongly — it can come across as too forceful or loud. "
                "Try relaxing slightly and letting your natural volume carry the conversation.", "yellow"
            )
        return (
            "Your voice isn't carrying as well as it could. "
            "Sit up straight, take a breath from your belly (not your chest), "
            "and speak as if the interviewer is sitting across a large table from you. "
            "More air behind your voice = more presence.", "red"
        )

    if "loudness_dynamics_power" in feat:
        if val < zone_min:
            return (
                "Your volume stays very flat throughout — there's not much variation between "
                "your soft and loud moments. This makes it harder to hold attention. "
                "Try deliberately saying certain phrases louder, and let your voice drop "
                "naturally at the end of a thought.", "red"
            )
        return (
            "Your volume dynamics could be more expressive. "
            "Think of your answer as a story — build up to key points and ease off after.", "yellow"
        )

    if "loudness_sma3_stddevNorm" in feat:
        if val < zone_min:
            return (
                "Your speaking volume is very consistent — which sounds controlled but can feel robotic. "
                "Let your voice naturally get a little louder when you're making a key point, "
                "and softer when you're wrapping up a thought.", "yellow"
            )
        if val > zone_max:
            return (
                "Your volume fluctuates quite a lot, which can be distracting. "
                "Try recording yourself and listening back — focus on keeping a steadier baseline volume.", "yellow"
            )

    if "loudness_sma3_percentile20.0" in feat:
        return (
            "Even your quietest moments are a bit too soft. "
            "Listeners shouldn't have to strain to catch words — keep your 'quiet' volume "
            "at about 70% of your normal speaking volume, not near a whisper.", "red"
        )

    if "loudness_sma3_percentile50.0" in feat or "loudness_sma3_percentile80.0" in feat:
        if feat_dir == "INCREASING" and val < zone_min:
            return (
                "Your overall speaking volume is on the low side. "
                "You don't need to shout — but projecting a little more "
                "makes you sound more confident and easier to follow.", "red"
            )
        return (
            "Your typical speaking volume could be adjusted slightly "
            "for a more confident, natural delivery.", "yellow"
        )

    if "loudness_sma3_pctlrange0-2" in feat:
        if val < zone_min:
            return (
                "The gap between your softest and loudest moments is very small. "
                "Widening this range — speaking softer in relaxed moments and louder "
                "at important ones — makes you sound much more dynamic and engaging.", "yellow"
            )

    if "loudness_sma3_meanRisingSlope" in feat:
        return (
            "You tend to start sentences softly and don't build much energy as you speak. "
            "Try starting each sentence with a little more intention — "
            "the first few words set the tone for the whole thought.", "yellow"
        )

    if "loudness_sma3_meanFallingSlope" in feat:
        return (
            "Your sentence endings trail off — the last few words become quiet and harder to catch. "
            "This can sound uncertain. Make sure the end of each sentence is as clear as the beginning.", "yellow"
        )

    if "loudness_sma3_stddevRisingSlope" in feat or "loudness_sma3_stddevFallingSlope" in feat:
        return (
            "Your volume build-ups and endings are uneven — sometimes you ramp up naturally, "
            "other times you don't. Aim for a more consistent rhythm: build into key points, "
            "land your ending, pause.", "yellow"
        )

    # ── PITCH & EXPRESSIVENESS ────────────────────────────────────────────────
    if "F0semitoneFrom27.5Hz_sma3nz_stddevNorm" in feat:
        if val < zone_min:
            return (
                "Your pitch is very flat — you're speaking in almost the same tone throughout. "
                "This is one of the most common reasons people sound robotic. "
                "Try this: when making your most important point, let your voice rise slightly. "
                "When concluding, let it naturally drop. Even small changes make a big difference.", "red"
            )
        if val > zone_max:
            return (
                "Your pitch swings quite a lot — it can sound a bit unsettled or over-animated. "
                "Aim for purposeful intonation: raise your pitch when introducing something new, "
                "lower it when making a confident statement.", "yellow"
            )
        return "Your pitch range could be slightly more expressive for a natural, engaging delivery.", "yellow"

    if "spectralFlux_sma3_amean" in feat:
        if val < zone_min:
            return (
                "Your voice sounds quite static — the tone doesn't change much as you speak. "
                "Expressive speakers let their voice shift as they move through ideas. "
                "Try reading a paragraph aloud and intentionally changing your tone "
                "between sentences — make it feel like a conversation.", "red"
            )
        return (
            "Your tonal variety could be more expressive. "
            "Let your voice naturally rise and fall as you explain different ideas.", "yellow"
        )

    if "spectralFlux_sma3_stddevNorm" in feat:
        return (
            "Your tonal consistency could be improved. "
            "Aim for expressive but smooth shifts in tone — not sudden jumps.", "yellow"
        )

    if "brightness_contrast" in feat:
        if val < zone_min:
            return (
                "Your voice lacks crispness and forward clarity. "
                "This can make you sound a bit muffled or distant. "
                "Try: open your mouth slightly more as you speak, "
                "and direct your voice towards your listener rather than letting it drop downward.", "yellow"
            )
        if val > zone_max:
            return (
                "Your voice sounds a bit harsh or shrill. "
                "Try relaxing your jaw and throat slightly as you speak — "
                "a warmer, more relaxed tone comes across as more natural and confident.", "yellow"
            )

    if "alphaRatioV_sma3nz_amean" in feat:
        if val < zone_min:
            return (
                "Your voice sounds a bit dull or muffled — it lacks brightness. "
                "Think about speaking more 'forward' — teeth slightly apart, "
                "lips more active, like you're speaking towards the front of your mouth.", "yellow"
            )
        return "Your voice brightness could be adjusted for a cleaner, crisper sound.", "yellow"

    if "alphaRatioV_sma3nz_stddevNorm" in feat:
        return (
            "Your voice brightness is inconsistent — some parts sound crisp, others sound dull. "
            "Try to maintain the same forward, open articulation throughout your answer.", "yellow"
        )

    if "hammarbergIndexV_sma3nz_stddevNorm" in feat:
        return (
            "Your vocal texture shifts unevenly — this can sound a little scattered. "
            "Focus on keeping your breath support steady throughout each sentence.", "yellow"
        )

    if "slopeV0-500_sma3nz_stddevNorm" in feat:
        return (
            "Your bass tone varies quite a bit, which can make your voice feel unstable. "
            "Try maintaining a consistent, grounded speaking tone — "
            "especially at the start and end of sentences.", "yellow"
        )

    # ── VOICE QUALITY ─────────────────────────────────────────────────────────
    if "HNRdBACF_sma3nz_amean" in feat:
        if val < zone_min:
            return (
                "Your voice sounds breathy or slightly rough — "
                "there's some noise mixed in with the tone. "
                "This often comes from speaking without enough breath support. "
                "Before you answer, take a slow breath from your belly. "
                "Also try humming quietly for 10 seconds to warm up your voice.", "red"
            )
        return "Your voice clarity could be slightly improved — try warming up before your interview.", "yellow"

    if "HNRdBACF_sma3nz_stddevNorm" in feat:
        return (
            "Your voice smoothness is inconsistent — clear at times, slightly rough at others. "
            "Staying hydrated and taking steady breaths between sentences will help.", "yellow"
        )

    if "shimmerLocaldB_sma3nz_amean" in feat:
        if val > zone_max:
            return (
                "Your voice is wobbling slightly in volume from word to word. "
                "This can sound shaky or nervous. "
                "Focus on steady breath support — breathe from your belly and "
                "keep your core gently engaged as you speak. "
                "Practice: sustain an 'ahhh' sound for 5 seconds at one volume.", "red"
            )
        if val < zone_min:
            return (
                "Your voice sounds very rigid — there's almost no natural vibration. "
                "This can sound a bit flat or robotic. "
                "Try loosening your jaw and throat — speak as if you're having a natural conversation.", "yellow"
            )
        return "Your voice steadiness could be slightly improved.", "yellow"

    if "shimmerLocaldB_sma3nz_stddevNorm" in feat:
        if val > zone_max:
            return (
                "Your voice wobble is very unpredictable — it changes a lot between moments. "
                "This often means your breath support is inconsistent. "
                "Take controlled breaths and keep your posture upright while speaking.", "red"
            )
        return (
            "Your voice steadiness could be more consistent. "
            "Focus on keeping a steady airflow as you move through longer sentences.", "yellow"
        )

    if "vocal_instability" in feat:
        if val > zone_max:
            return (
                "Your voice is showing signs of tension — it's both a little shaky and unsteady. "
                "This often happens when we're nervous. "
                "Try a slow exhale before answering, relax your shoulders, "
                "and let your voice sit lower in your chest. "
                "Stability comes from breath, not from forcing your voice to sound calm.", "red"
            )
        if val < zone_min:
            return (
                "Your voice sounds almost too perfectly still — which can feel mechanical. "
                "Natural speech has a small amount of variation. "
                "Try speaking more casually and less 'performed'.", "yellow"
            )
        return "Your overall vocal stability could be slightly improved.", "yellow"

    if "F2amplitudeLogRelF0_sma3nz_amean" in feat:
        if val < zone_min:
            return (
                "Your vowels sound a bit weak or swallowed. "
                "Open your mouth slightly more when saying vowels — "
                "think 'ay', 'ee', 'eye', 'oh', 'oo' with a fuller sound. "
                "Strong vowels give your voice body and presence.", "yellow"
            )
        if val > zone_max:
            return (
                "Your vowels are a bit overpowering — it can sound forced. "
                "Speak naturally without exaggerating vowel sounds.", "yellow"
            )

    if "F2amplitudeLogRelF0_sma3nz_stddevNorm" in feat:
        return (
            "Your vowel strength is inconsistent — some words sound full, others sound thin. "
            "Focus on consistent mouth opening throughout your answer.", "yellow"
        )

    if "F3amplitudeLogRelF0_sma3nz_amean" in feat:
        if val < zone_min:
            return (
                "Your voice lacks natural depth and richness. "
                "Try speaking from slightly lower in your throat — "
                "not forced, just relaxed and resonant. "
                "Imagine your voice filling a room rather than just reaching across a desk.", "yellow"
            )
        if val > zone_max:
            return (
                "Your voice is very rich or resonant — which is good — but can occasionally "
                "sound over-produced. A slightly lighter, more natural tone works better in interviews.", "yellow"
            )

    if "F3amplitudeLogRelF0_sma3nz_stddevNorm" in feat:
        return (
            "Your vocal richness changes quite a bit throughout — sometimes full, sometimes thin. "
            "Try to maintain consistent resonance, especially through longer answers.", "yellow"
        )

    if "mfcc1_sma3_stddevNorm" in feat or "mfcc1V_sma3nz_stddevNorm" in feat:
        if val < zone_min:
            return (
                "Your voice texture is quite uniform — it doesn't change much. "
                "Varying your articulation (how you shape words) adds richness. "
                "Try over-pronouncing consonants slightly for 30 seconds, "
                "then speaking normally — you'll notice more texture.", "yellow"
            )
        if val > zone_max:
            return (
                "Your voice texture changes very abruptly — it can sound disjointed. "
                "Focus on smooth, consistent articulation from word to word.", "yellow"
            )

    if "mfcc2V_sma3nz_stddevNorm" in feat:
        if val > zone_max:
            return (
                "The warmth in your voice fluctuates too much — "
                "some phrases sound warm and natural, others sound cold or strained. "
                "Keep your throat relaxed and your breathing steady throughout.", "yellow"
            )
        if val < zone_min:
            return (
                "Your vocal warmth is very flat — your voice may sound a bit cold or robotic. "
                "Try thinking of something you're genuinely interested in before answering — "
                "warmth in voice often follows warmth in mindset.", "yellow"
            )

    if "mfcc3_sma3_stddevNorm" in feat:
        if val < zone_min:
            return (
                "Your voice has very little tonal colour — it sounds one-dimensional. "
                "Let your voice reflect the meaning of what you're saying: "
                "slightly brighter when enthusiastic, warmer when reflective.", "yellow"
            )
        if val > zone_max:
            return (
                "Your tonal colour shifts quite erratically. "
                "Aim for smooth, intentional variation rather than sudden changes.", "yellow"
            )

    # ── FLUENCY & FLOW ────────────────────────────────────────────────────────
    if "voiced_flow" in feat:
        if val < zone_min:
            return (
                "Your speech feels choppy — you're breaking ideas into too many small fragments. "
                "Before answering, structure your response as: "
                "Point → Reason → Example. Then say each part in one connected breath. "
                "This gives your speech a natural, flowing rhythm.", "red"
            )
        if val > zone_max:
            return (
                "You're speaking in very long, unbroken stretches without natural pauses. "
                "Pause briefly after each key point — it gives listeners time to absorb what you said "
                "and makes you sound more deliberate and in control.", "yellow"
            )
        return "Your speech flow could be slightly more connected and rhythmic.", "yellow"

    if "MeanUnvoicedSegmentLength" in feat:
        if val > zone_max:
            return (
                "Your pauses are too long — they break your momentum and can make you seem uncertain. "
                "Aim for pauses of about 1 second between ideas. "
                "If you need thinking time, use a filler phrase like 'That's a good point — ' "
                "rather than a long silence.", "red"
            )
        if val < zone_min:
            return (
                "Your pauses are very short — you're rushing between thoughts. "
                "Allow a brief moment of silence after your key points. "
                "Silence isn't weakness — it signals confidence and lets your words sink in.", "yellow"
            )
        return "Your pause timing could be better calibrated for a more natural rhythm.", "yellow"

    # ── Generic fallback (direction-aware) ───────────────────────────────────
    if val < zone_min:
        if feat_dir == "INCREASING":
            return (
                f"Your {friendly} is below the ideal range. "
                "Try recording yourself and listening back to identify what feels off.", "red"
            )
        elif feat_dir == "DECREASING":
            return (
                f"Your {friendly} is lower than the ideal range. "
                "You've actually reduced this too much — a slightly higher value would hit the sweet spot.", "yellow"
            )
        else:  # MIDDLE_IS_BEST
            return (
                f"Your {friendly} is a bit low — nudging it slightly upward will put you "
                "right in the ideal range.", "yellow"
            )
    if val > zone_max:
        if feat_dir == "DECREASING":
            return (
                f"Your {friendly} is higher than the ideal range — "
                "bringing it down a little will help your delivery sound more natural.", "red"
            )
        elif feat_dir == "INCREASING":
            return (
                f"Your {friendly} is a bit high — "
                "dialling it back slightly will land you right in the elite range.", "yellow"
            )
        else:  # MIDDLE_IS_BEST
            return (
                f"Your {friendly} is a bit high — pulling it back slightly will hit "
                "the sweet spot for a natural delivery.", "yellow"
            )
    return f"Your {friendly} could be fine-tuned slightly for a more polished delivery.", "yellow"


# ── Eligible feature selector ─────────────────────────────────────────────────
def _get_eligible_features(
    shap_dict: dict,
    group_members: list,
    feature_values: dict,
    direction: str,
    overall_score: float,
    golden_zones: dict,
) -> list:
    if direction == "negative":
        if overall_score < 0.65:
            sensitivity = 0.001
        elif overall_score > 0.80:
            sensitivity = 0.015
        else:
            sensitivity = 0.005
    else:
        sensitivity = 0.003

    eligible = []
    for feat, shap_val in shap_dict.items():
        if feat not in group_members:
            continue
        if feat in SKIP_COACHING:
            continue

        val    = feature_values.get(feat, 0)
        zone   = golden_zones.get(feat, {"min": -99, "max": 99})
        in_zone = zone.get("min", -99) <= val <= zone.get("max", 99)

        if direction == "positive":
            if shap_val <= sensitivity:
                continue
        else:
            if shap_val >= -sensitivity:
                continue
            if overall_score > 0.80 and in_zone:
                continue

        eligible.append((feat, shap_val))

    eligible.sort(key=lambda x: abs(x[1]), reverse=True)
    return eligible


# ── Group results builder ─────────────────────────────────────────────────────
def build_audio_group_results(
    shap_dict: dict,
    feature_values: dict,
    overall_score: float,
    base_value: float,
    golden_zones: dict,
) -> dict:
    bv = float(base_value[0]) if isinstance(base_value, (list, np.ndarray)) else float(base_value)
    baseline_pct         = bv * 100
    explainer_prediction = baseline_pct
    group_results        = {}

    for group_name, group_info in FEEDBACK_GROUPS.items():
        members = group_info["members"]

        group_shap_sum = sum(v for f, v in shap_dict.items() if f in members)
        impact_points  = group_shap_sum * 100
        explainer_prediction += impact_points

        if   group_shap_sum >= group_info["yellow_threshold"]: status = "green"
        elif group_shap_sum >= group_info["red_threshold"]:    status = "yellow"
        else:                                                  status = "red"

        neg_features = _get_eligible_features(
            shap_dict, members, feature_values, "negative", overall_score, golden_zones
        )
        pos_features = _get_eligible_features(
            shap_dict, members, feature_values, "positive", overall_score, golden_zones
        )

        if   status == "green":  p_limit, n_limit = 3, 1
        elif status == "yellow": p_limit, n_limit = 2, 2
        else:                    p_limit, n_limit = 1, 3

        tips: list = []
        seen: set  = set()

        def process_tips(features, dir_type, limit):
            count = 0
            for feat, shap_val in features:
                if count >= limit or feat in seen:
                    continue
                val  = feature_values.get(feat, 0)
                # zone comes directly from calibration JSON — no overrides applied
                zone = golden_zones.get(feat, {})
                tip_text, tip_status = _get_coaching_tip(feat, val, zone, dir_type)
                tips.append({
                    "feature":        feat,
                    "friendly":       FRIENDLY_NAMES.get(feat, feat),
                    "shap":           round(float(shap_val), 4),
                    "value":          round(float(val), 4),
                    "direction":      dir_type,
                    "tip":            tip_text,
                    "status":         tip_status,
                    "zone_min":       round(float(zone.get("min", -99)), 4),
                    "zone_max":       round(float(zone.get("max",  99)), 4),
                    # zone_direction is ground truth from JSON — frontend reads
                    # this to show "More is better / Less is better / Middle is best"
                    "zone_direction": zone.get("direction", "INCREASING"),
                })
                seen.add(feat)
                count += 1

        if status == "red":
            process_tips(neg_features, "negative", n_limit)
            process_tips(pos_features, "positive", p_limit)
        else:
            process_tips(pos_features, "positive", p_limit)
            process_tips(neg_features, "negative", n_limit)

        group_results[group_name] = {
            "impact_points": round(float(impact_points), 1),
            "status":        status,
            "tips":          tips,
        }

    group_results["_metadata"] = {
        "baseline":    round(baseline_pct, 1),
        "synergy_gap": round((overall_score * 100) - explainer_prediction, 1),
    }
    return group_results


# ============================================================
# LEGACY COACHING KNOWLEDGE BASE
# ============================================================
FEATURE_LABELS = {
    "loudnessPeaksPerSec":                        "Emphasis & Stress",
    "loudness_sma3_percentile20.0":               "Quietest Moments",
    "loudness_dynamics_power":                    "Volume Dynamics",
    "vocal_projection":                           "How Well Your Voice Carries",
    "loudness_sma3_meanRisingSlope":              "Sentence Energy Build-up",
    "loudness_sma3_stddevNorm":                   "Volume Variety",
    "loudness_sma3_pctlrange0-2":                 "Quiet-to-Loud Range",
    "loudness_sma3_meanFallingSlope":             "Sentence Endings",
    "loudness_sma3_stddevRisingSlope":            "Energy Consistency",
    "voiced_flow":                                "Speech Flow",
    "vocal_instability":                          "Overall Vocal Stability",
    "shimmerLocaldB_sma3nz_stddevNorm":           "Voice Steadiness Consistency",
    "HNRdBACF_sma3nz_amean":                      "Voice Smoothness",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":     "Pitch Range",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm":      "Voice Richness Variation",
    "F3amplitudeLogRelF0_sma3nz_amean":           "Voice Richness",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm":      "Vowel Strength Variation",
    "F2amplitudeLogRelF0_sma3nz_amean":           "Vowel Strength",
    "mfcc1_sma3_stddevNorm":                      "Voice Texture Variety",
    "mfcc2V_sma3nz_stddevNorm":                   "Vocal Warmth Variation",
    "mfcc3_sma3_stddevNorm":                      "Tonal Color Variety",
    "hammarbergIndexV_sma3nz_stddevNorm":         "Vocal Texture Consistency",
    "alphaRatioV_sma3nz_amean":                   "Voice Brightness",
    "alphaRatioV_sma3nz_stddevNorm":              "Voice Brightness Consistency",
    "slopeV0-500_sma3nz_stddevNorm":              "Bass Tone Variation",
    "spectralFlux_sma3_amean":                    "Tonal Variety",
    "MeanUnvoicedSegmentLength":                  "Pause Length",
    "brightness_contrast":                        "Voice Sharpness & Clarity",
}

FEATURE_TIPS = {
    "loudnessPeaksPerSec": {
        "positive": "Great — you naturally stress key words, which keeps listeners engaged.",
        "negative": (
            "Your speaking volume is very flat — everything sounds equally important, "
            "which makes nothing stand out. Pick the 2–3 most important words in each sentence "
            "and say them noticeably louder. Practice this on one sentence at a time."
        ),
    },
    "vocal_projection": {
        "positive": "Your voice carries well — you sound confident and easy to hear.",
        "negative": (
            "Your voice isn't projecting enough. Sit up straight, breathe from your belly "
            "rather than your chest, and imagine speaking to someone sitting across a large table. "
            "More air behind your voice means more presence."
        ),
    },
    "vocal_instability": {
        "positive": "Your voice stays steady — you sound calm and in control.",
        "negative": (
            "Your voice sounds a bit tense or shaky. Before speaking, exhale slowly "
            "for 5 seconds, relax your shoulders, and let your voice sit lower in your chest. "
            "Stability comes from breath — not from forcing yourself to sound calm."
        ),
    },
    "loudness_dynamics_power": {
        "positive": "Good variation in how loud and soft you get — you sound engaging and natural.",
        "negative": (
            "Your volume stays flat throughout. Try this: say a sentence normally, "
            "then say the most important word in it noticeably louder. "
            "Do this for 10 sentences a day."
        ),
    },
    "loudness_sma3_meanRisingSlope": {
        "positive": "You build energy into sentences well — your openings sound confident.",
        "negative": (
            "You tend to start sentences softly. The first few words set the tone for the whole thought — "
            "try starting each sentence with a little more intention and energy."
        ),
    },
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": {
        "positive": "Your pitch rises and falls naturally — you sound expressive and engaged.",
        "negative": (
            "Your pitch is very flat — you're speaking in almost the same tone throughout. "
            "This is one of the most common reasons people sound robotic. "
            "Let your voice rise when introducing a key point and drop naturally at the end. "
            "Even small changes make a big difference."
        ),
    },
    "voiced_flow": {
        "positive": "Your speech flows smoothly — ideas connect naturally.",
        "negative": (
            "Your speech feels choppy. Before answering, structure your response as: "
            "Point → Reason → Example. Then say each part in one connected breath. "
            "This creates a natural, flowing rhythm."
        ),
    },
    "loudness_sma3_percentile20.0": {
        "positive": "Even your quietest moments are clear and audible.",
        "negative": (
            "Your quietest moments are too soft — listeners may miss words. "
            "Think of your 'quiet' speaking volume as 70% of your normal voice, not near a whisper."
        ),
    },
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": {
        "positive": "Your voice has good depth and richness — it's pleasant to listen to.",
        "negative": (
            "Your voice sounds a bit thin. Try speaking from slightly lower in your throat — "
            "relaxed and resonant, not forced. Imagine your voice filling the room."
        ),
    },
    "F2amplitudeLogRelF0_sma3nz_amean": {
        "positive": "Your vowels are strong and clear — great for intelligibility.",
        "negative": (
            "Your vowels sound a bit weak or swallowed. "
            "Open your mouth slightly more when speaking — especially on vowel sounds. "
            "Strong vowels give your voice body and presence."
        ),
    },
    "HNRdBACF_sma3nz_amean": {
        "positive": "Your voice sounds clean and smooth — easy and pleasant to listen to.",
        "negative": (
            "Your voice sounds a bit breathy or rough. "
            "This often comes from speaking without enough breath support. "
            "Take a slow belly breath before answering, and try humming for 10 seconds to warm up."
        ),
    },
    "shimmerLocaldB_sma3nz_stddevNorm": {
        "positive": "Your voice stays steady — you sound composed and consistent.",
        "negative": (
            "Your voice wobbles unpredictably from word to word. "
            "Practice sustaining an 'ahhh' sound at one volume for 5 seconds — "
            "this trains steadier breath control."
        ),
    },
    "MeanUnvoicedSegmentLength": {
        "positive": "Your pauses are well-timed — they feel deliberate, not hesitant.",
        "negative": (
            "Your pauses are too long — they break your momentum and can make you seem uncertain. "
            "Aim for about 1 second between ideas. "
            "If you need thinking time, use a brief filler phrase rather than a long silence."
        ),
    },
    "mfcc2V_sma3nz_stddevNorm": {
        "positive": "Your vocal warmth varies naturally — you sound human and engaging.",
        "negative": (
            "The warmth in your voice fluctuates too much — some phrases sound natural, "
            "others sound strained. Keep your throat relaxed and your breathing steady throughout."
        ),
    },
    "alphaRatioV_sma3nz_amean": {
        "positive": "Your voice sounds clear and forward — it's easy to understand.",
        "negative": (
            "Your voice sounds a bit muffled or dull. "
            "Try speaking more 'forward' — teeth slightly apart, lips more active, "
            "as if directing your voice towards your listener."
        ),
    },
}


def _get_generic_tip(label: str, is_positive: bool) -> str:
    if is_positive:
        return f"Your {label.lower()} is a real strength — keep doing what you're doing."
    return (
        f"Your {label.lower()} could be improved. "
        "Try recording yourself speaking and listening back — "
        "you'll often notice patterns you can't hear in the moment."
    )


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("VOICE ANALYSIS SERVICE STARTING...")
    print("="*60)
    try:
        print("Loading models and initializing tools...")

        models['final_model']  = joblib.load("models/xgboost_balance_model.pkl")
        models['top_features'] = joblib.load("models/top_features_balance.pkl")
        print(f"   XGBoost model loaded — {len(models['top_features'])} features")

        models['smile'] = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("   openSMILE eGeMAPSv02 initialized")

        models['shap_proxy']    = joblib.load("models/shap_proxy_calibrated.pkl")
        models['shap_explainer'] = shap.TreeExplainer(models['shap_proxy'])
        print("   SHAP TreeExplainer ready (calibration-aware proxy)")

        models['beta_params'] = joblib.load("models/beta_params_production.pkl")
        bp = models['beta_params']
        print(f"   Beta calibrator loaded — params {bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}")

        if not Path(CALIBRATION_PATH).exists():
            raise FileNotFoundError(f"Missing calibration file: {CALIBRATION_PATH}.")
        with open(CALIBRATION_PATH) as f:
            calibration = json.load(f)
        models['golden_zones'] = calibration["golden_zones"]
        models['global_stats'] = calibration.get("global_stats", {})
        print(f"   Audio golden zones loaded: {len(models['golden_zones'])} features")

        models['shap_features'] = list(models['shap_proxy'].feature_names_in_)

        print("\nSERVICE READY!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\nStartup failed: {e}")
        raise

    yield
    print("\nVoice analysis service shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Hirely Voice Analysis Service",
    description="Analyzes candidate voice confidence from interview audio",
    lifespan=lifespan
)

try:
    redis_client = Redis(
        url=os.getenv("REDIS_URL"),
        token=os.getenv("REDIS_TOKEN"),
    )
    redis_client.ping()
    print("Upstash Redis connected")
except Exception as e:
    print(f"Upstash Redis connection failed: {e}")
    redis_client = None


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_opensmile_features(filepath: str) -> pd.DataFrame:
    fpath = Path(filepath)
    if not fpath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    logger.info(f"Extracting openSMILE features: {fpath.name}")

    try:
        feats = models['smile'].process_file(str(fpath))
    except Exception as e:
        logger.warning(f"Direct openSMILE processing failed ({e}). Converting to WAV...")
        tmp_path = None
        try:
            tmp_path = tempfile.mktemp(suffix=".wav")
            audio = AudioSegment.from_file(str(fpath))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp_path, format="wav")
            feats = models['smile'].process_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    return pd.DataFrame([feats.iloc[0].to_dict()])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()

    if all(c in cols for c in ['jitterLocal_sma3nz_amean', 'shimmerLocaldB_sma3nz_amean']):
        df['vocal_instability'] = (
            df['jitterLocal_sma3nz_amean'] * df['shimmerLocaldB_sma3nz_amean']
        )

    if all(c in cols for c in ['loudness_sma3_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm']):
        df['vocal_projection'] = (
            df['loudness_sma3_amean'] / (df['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] + 0.05)
        )

    if all(c in cols for c in ['HNRdBACF_sma3nz_amean', 'alphaRatioV_sma3nz_amean']):
        df['voice_quality'] = df['HNRdBACF_sma3nz_amean'] - df['alphaRatioV_sma3nz_amean']

    if all(c in cols for c in ['loudness_sma3_pctlrange0-2', 'loudnessPeaksPerSec']):
        df['loudness_dynamics_power'] = (
            df['loudness_sma3_pctlrange0-2'] * df['loudnessPeaksPerSec']
        )

    if all(c in cols for c in ['VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec']):
        df['voiced_flow'] = (
            df['VoicedSegmentsPerSec'] * df['MeanVoicedSegmentLengthSec']
        )

    if all(c in cols for c in ['slopeV500-1500_sma3nz_amean', 'slopeV0-500_sma3nz_amean']):
        df['brightness_contrast'] = (
            df['slopeV500-1500_sma3nz_amean'] - df['slopeV0-500_sma3nz_amean']
        )

    return df


def compute_wpm(audio_path: str, transcript_text: str | None) -> float:
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000.0
        if transcript_text and transcript_text.strip() and duration_sec > 0:
            word_count = len(transcript_text.strip().split())
            return round((word_count / duration_sec) * 60, 1)
        return 0.0
    except Exception:
        return 0.0


# ============================================================
# SCORING
# ============================================================
def beta_calibrate(raw_score: float) -> float:
    c, d, e = [float(p) for p in models['beta_params']]
    x = float(np.clip(raw_score, 1e-5, 1.0 - 1e-5))
    log_odds = c * np.log(x) + d * np.log(1.0 - x) + e
    return float(1.0 / (1.0 + np.exp(-log_odds)))


def score_to_label(score: float) -> str:
    if score >= 0.75:  return "Highly Confident"
    if score >= 0.50:  return "Confident"
    if score >= 0.30:  return "Moderately Confident"
    return "Needs Improvement"


# ============================================================
# SHAP ENGINE
# ============================================================
def generate_shap_and_groups(
    X: pd.DataFrame,
    calibrated_score: float,
    feature_values_dict: dict,
) -> dict:
    try:
        explainer     = models['shap_explainer']
        shap_features = models['shap_features']
        golden_zones  = models['golden_zones']

        X_shap = pd.DataFrame(index=[0], columns=shap_features, dtype=float)
        for col in shap_features:
            X_shap[col] = feature_values_dict.get(col, 0.0)

        X_np      = X_shap.to_numpy(dtype=np.float64)
        shap_vals = explainer.shap_values(X_np)[0]
        base_value = float(explainer.expected_value)

        shap_dict = {feat: float(sv) for feat, sv in zip(shap_features, shap_vals)}

        group_results = build_audio_group_results(
            shap_dict      = shap_dict,
            feature_values = feature_values_dict,
            overall_score  = calibrated_score,
            base_value     = base_value,
            golden_zones   = golden_zones,
        )

        top_contributors = []
        for feat, sv in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(sv) < 0.008:
                continue
            fval      = feature_values_dict.get(feat, 0.0)
            label     = FEATURE_LABELS.get(feat, FRIENDLY_NAMES.get(feat, feat))
            direction = "positive" if sv > 0 else "negative"
            tip       = FEATURE_TIPS.get(feat, {}).get(direction, _get_generic_tip(label, sv > 0))
            top_contributors.append({
                "feature":          feat,
                "label":            label,
                "value":            round(float(fval), 4),
                "shap_value":       round(sv, 4),
                "direction":        "increased" if sv > 0 else "decreased",
                "impact_magnitude": round(abs(sv), 4),
                "explanation":      tip,
            })

        neg_contributors = [c for c in top_contributors if c["direction"] == "decreased"]
        pos_contributors = [c for c in top_contributors if c["direction"] == "increased"]

        _max_improvements = (
            1 if calibrated_score >= 0.70 else
            2 if calibrated_score >= 0.50 else
            len(neg_contributors)
        )

        return {
            "group_results":     group_results,
            "shap_dict":         shap_dict,
            "base_value":        round(base_value, 4),
            "top_contributors":  top_contributors,
            "top_improvements":  neg_contributors[:_max_improvements],
            "top_strengths":     pos_contributors[:3],
            "all_shap_values":   {f: round(v, 4) for f, v in shap_dict.items()},
        }

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return {
            "group_results":    {},
            "shap_dict":        {},
            "base_value":       0.0,
            "top_contributors": [],
            "top_improvements": [],
            "top_strengths":    [],
            "all_shap_values":  {},
        }


# ============================================================
# PREDICTION PIPELINE
# ============================================================
def predict_confidence(audio_path: str, transcript_text: str | None = None) -> dict:
    logger.info(f"Starting voice analysis: {audio_path}")

    features_df = extract_opensmile_features(audio_path)
    features_df = add_engineered_features(features_df)

    top_features = models['top_features']
    X = pd.DataFrame(index=[0], columns=top_features, dtype=float)
    for col in top_features:
        X[col] = features_df[col].iloc[0] if col in features_df.columns else 0.0

    raw_score        = float(np.clip(models['final_model'].predict(X)[0], 0.0, 1.0))
    calibrated_score = beta_calibrate(raw_score)
    logger.info(f"Score: raw={raw_score:.4f} → calibrated={calibrated_score:.4f}")

    feature_values_dict = features_df.iloc[0].to_dict()
    for col in top_features:
        if col not in feature_values_dict:
            feature_values_dict[col] = float(X[col].iloc[0])

    shap_result = generate_shap_and_groups(X, calibrated_score, feature_values_dict)
    wpm = compute_wpm(audio_path, transcript_text)

    return {
        "confidence_score": calibrated_score,
        "raw_score":        raw_score,
        "confidence_label": score_to_label(calibrated_score),
        "wpm":              wpm,
        "shap":             shap_result,
        "raw_features":     features_df.iloc[0].to_dict(),
    }


def generate_final_summary(score: float, neg_features: list, pos_features: list) -> dict:
    if score > 0.6:
        opening = "You have a strong vocal presence — you sound ready for a high-level pitch."
    elif score > 0.45:
        opening = "Solid performance. You come across as professional and capable."
    else:
        opening = "A good start! A few focused tweaks to your delivery can really boost your impact."

    focus_note = best_trait = None
    if neg_features:
        main_fix_label = FEATURE_LABELS.get(neg_features[0].get("feature",""), "vocal delivery")
        focus_note = (
            f"Don't stress too much about the {main_fix_label.lower()} right now — "
            "every voice has its own natural patterns that show up under analysis."
        )
        if pos_features:
            best_trait_label = FEATURE_LABELS.get(pos_features[0].get("feature",""), "strengths")
            best_trait = (
                f"Focus on leaning into your {best_trait_label.upper()} — "
                "it's already working well for you."
            )

    return {
        "opening":    opening,
        "focus_note": focus_note,
        "best_trait": best_trait,
        "reminder": (
            "This is a simulation. Great interviewers look for your passion and thinking — "
            "this AI is just here to help you polish your delivery!"
        ),
    }


# ============================================================
# BACKGROUND TASK
# ============================================================
async def process_voice_analysis(
    turn_id: int,
    audio_path: str,
    interview_id: str,
    start_time: datetime,
    transcript_text: str | None = None,
):
    try:
        print(f"\n{'='*60}")
        print(f"VOICE ANALYSIS — Turn {turn_id} | Session {interview_id}")
        print(f"{'='*60}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        prediction = predict_confidence(audio_path, transcript_text)
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        raw           = prediction["raw_features"]
        shap          = prediction["shap"]
        group_results = shap["group_results"]

        vsps               = raw.get("VoicedSegmentsPerSec", 1.0) or 1.0
        mvsl               = raw.get("MeanVoicedSegmentLengthSec", 0.3) or 0.3
        voiced_fraction    = min(1.0, float(vsps) * float(mvsl))
        pause_ratio_approx = max(0.0, round(1.0 - voiced_fraction, 4))

        result = {
            "interviewTurnId": turn_id,
            "confidenceLevel":     round(prediction["confidence_score"], 4),
            "confidenceLabelText": prediction["confidence_label"],
            "speakingQuality":  round(prediction["confidence_score"], 4),
            "vocalStability":   round(max(0.0, 1.0 - float(raw.get("jitterLocal_sma3nz_amean", 0) or 0)), 4),
            "speakingFluency":  round(voiced_fraction, 4),
            "pitchMean":        round(float(raw.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0) or 0), 4),
            "pitchStd":         round(float(raw.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0) or 0), 4),
            "energyLevel":      round(float(raw.get("equivalentSoundLevel_dBp", 0) or 0), 4),
            "wordsPerMinute":   prediction["wpm"],
            "pauseRatio":       pause_ratio_approx,
            "jitter":           round(float(raw.get("jitterLocal_sma3nz_amean", 0) or 0), 6),
            "shimmer":          round(float(raw.get("shimmerLocaldB_sma3nz_amean", 0) or 0), 6),
            "modelVersion": "v2.2-egemaps-shap-betacal-zones",
            "allProbabilities": shap["all_shap_values"],
            "groupResults": group_results,
            "rawFeatures": {
                **{
                    k: (round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in raw.items()
                },
                "groupResults": group_results,
                "shapExplanations": shap["top_contributors"],
                "shapBaseValue":    shap["base_value"],
                "featureExplanations": [
                    c["explanation"]
                    for c in shap["top_improvements"]
                    if c.get("explanation")
                ],
                "finalSummary": generate_final_summary(
                    prediction["confidence_score"],
                    shap["top_improvements"],
                    shap["top_strengths"],
                ),
            },
            "status":           "completed",
            "processingTimeMs": elapsed_ms,
            "processedAt":      datetime.now().isoformat(),
        }

        print(f"Confidence  : {result['confidenceLevel']*100:.1f}% ({result['confidenceLabelText']})")
        print(f"WPM         : {result['wordsPerMinute']}")
        print(f"Group statuses:")
        for gname, gdata in group_results.items():
            if gname == "_metadata":
                continue
            icon = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(gdata["status"],"?")
            print(f"  {icon} {gname}: {gdata['impact_points']:+.1f}%  ({len(gdata['tips'])} tips)")
        print(f"Processing  : {elapsed_ms}ms")
        print(f"{'='*60}\n")

        if redis_client:
            try:
                redis_client.set(
                    f"voice_analysis:{interview_id}:{turn_id}",
                    json.dumps(result),
                    ex=86400,
                )
                logger.info(f"Saved to Redis: voice_analysis:{interview_id}:{turn_id}")
            except Exception as e:
                logger.warning(f"Redis save failed (non-critical): {e}")

        return result

    except Exception as e:
        print(f"\nVOICE ANALYSIS FAILED — Turn {turn_id}: {e}\n")
        logger.error(f"process_voice_analysis error: {e}")

        error_result = {
            "interviewTurnId": turn_id,
            "status":          "failed",
            "errorMessage":    str(e),
            "processedAt":     datetime.now().isoformat(),
        }
        if redis_client:
            try:
                redis_client.set(
                    f"voice_analysis:{interview_id}:{turn_id}",
                    json.dumps(error_result),
                    ex=86400,
                )
            except Exception:
                pass


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "service":      "Hirely Voice Analysis v2.2",
        "models_loaded": bool(models.get("final_model")),
        "shap_ready":   bool(models.get("shap_explainer")),
        "zones_loaded": bool(models.get("golden_zones")),
        "timestamp":    datetime.now().isoformat(),
    }


class AnalysisRequest(BaseModel):
    turn_id:      int
    interview_id: str
    audio_path:   str
    transcript:   str | None = None


@app.post("/analyze-voice")
async def analyze_voice(request_data: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"Queued voice analysis — Turn {request_data.turn_id}")
        background_tasks.add_task(
            process_voice_analysis,
            request_data.turn_id,
            request_data.audio_path,
            request_data.interview_id,
            datetime.now(),
            request_data.transcript,
        )
        return {
            "status":    "queued",
            "turn_id":   request_data.turn_id,
            "message":   "Voice analysis queued.",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return {"status": "error", "error": str(e)}, 500


@app.get("/result/{interview_id}/{turn_id}")
def get_result(interview_id: str, turn_id: int):
    try:
        if not redis_client:
            return {"status": "error", "error": "Redis not connected"}
        result = redis_client.get(f"voice_analysis:{interview_id}:{turn_id}")
        if result is None:
            return {"status": "pending"}
        return json.loads(result) if isinstance(result, str) else result
    except Exception as e:
        logger.error(f"Get result failed: {e}")
        return {"status": "error", "error": str(e)}, 500


# ============================================================
# DEVELOPMENT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Hirely Voice Analysis Service v2.2")
    print("Docs: http://localhost:8001/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)