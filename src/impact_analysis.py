import os
import json
import logging
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini Client
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables.")
    client = None

# Fallback/Default composition data
STATIC_COMPOSITION: Dict[str, Dict[str, float]] = {
    "PCB": {"Gold": 1.0, "Copper": 10.0, "Aluminum": 15.0, "Plastic": 30.0, "Others": 44.0},
    "adapter": {"Gold": 0.1, "Copper": 15.0, "Aluminum": 5.0, "Plastic": 60.0, "Others": 19.9},
    "cable": {"Gold": 0.0, "Copper": 60.0, "Aluminum": 5.0, "Plastic": 30.0, "Others": 5.0},
    "laptop": {"Gold": 0.5, "Copper": 12.0, "Aluminum": 20.0, "Plastic": 35.0, "Others": 32.5},
    "mouse": {"Gold": 0.1, "Copper": 5.0, "Aluminum": 2.0, "Plastic": 80.0, "Others": 12.9},
    "smartphone": {"Gold": 1.0, "Copper": 10.0, "Aluminum": 15.0, "Plastic": 30.0, "Others": 44.0},
}

def get_ai_analysis(class_name: str) -> Dict[str, Any]:
    """
    Get dynamic material composition and impact info from Gemini.
    """
    if not client:
        return {"composition": STATIC_COMPOSITION.get(class_name, {}), "source": "static"}

    prompt = f"""
    Analyze the material composition and environmental impact of a single '{class_name}' for an e-waste project.
    Return ONLY a JSON object with this exact structure:
    {{
      "composition": {{
        "Gold": percentage_weight,
        "Copper": percentage_weight,
        "Aluminum": percentage_weight,
        "Plastic": percentage_weight,
        "Others": percentage_weight
      }},
      "impact": "A detailed 3-4 sentence paragraph. Explicitly estimate the actual weight (e.g., in milligrams or grams) of valuable metals like Gold, Copper, and Aluminum that can be extracted or recycled from a typical single '{class_name}'. Explain how extracting these materials helps the environment and contributes to the economy."
    }}
    Ensure the percentages in 'composition' add up to 100.
    """

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            # Clean response text in case of markdown blocks
            clean_text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_text)
            data["source"] = "ai"
            return data
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning(f"Rate limit hit during analysis. Retrying in {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            logger.error(f"Error calling Gemini API: {e}")
            break
    
    return {"composition": STATIC_COMPOSITION.get(class_name, {}), "source": "static_fallback"}

def get_bulk_impact_summary(detections_list: List[str]) -> str:
    """
    Get a global impact summary for a set of detected e-waste items.
    """
    if not client or not detections_list:
        return "Please recycle these electronic items at a formal facility to recover valuable metals and prevent toxic pollution."

    items_str = ", ".join(detections_list)
    prompt = f"""
    Based on the following detected e-waste items: {items_str}.
    Provide a detailed "AI Insight" focusing heavily on the specific environmental and economic impact of RECYCLING these exact objects.
    Format the response as a clear Markdown bulleted list with exactly 4 to 6 facts. Incorporate the following themes:
    - The staggering amount of THESE specific items generated globally and their footprint.
    - The precise economic value derived from recycling them (e.g., valuable metals like gold, silver, copper recovered compared to mined ore).
    - The profound environmental benefit of recycling them (e.g., preventing specific toxic substances like lead, mercury, or cadmium from leaching into the soil/water).
    - Provide quantifiable metrics or impressive statistics wherever possible.
    Make the facts punchy and use **bold text** for key numbers and concepts. **DO NOT USE ANY EMOJIS.** Do not include any introductory or concluding text, JUST the bullet points.
    """

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning(f"Rate limit hit during bulk summary. Retrying in {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            logger.error(f"Error calling Gemini API for bulk summary: {e}")
            break
            
    return "Properly recycling these items contributes to a circular economy and protects soil and water health."
