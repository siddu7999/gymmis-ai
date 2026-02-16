# ai_model.py — US staples + Indian basics, with CLIP/SigLIP re-rank
import os, uuid
from typing import Dict, Any
from PIL import Image
from transformers import pipeline

# --------- Env config ---------
HF_TOKEN        = os.getenv("HF_TOKEN", None)
HF_MODEL_GENERAL= os.getenv("HF_MODEL_GENERAL", "Jacques7103/Food-Recognition")
HF_MODEL_INDIAN = os.getenv("HF_MODEL_INDIAN",  "Utsav201247/food_recognition")   # optional
HF_MODEL_ZS     = os.getenv("HF_MODEL_ZS",     "openai/clip-vit-base-patch32")    # or "google/siglip-so400m-patch14-384" if available

TOP_K      = int(os.getenv("FOOD_TOP_K", "5"))
ZS_TOP_K   = int(os.getenv("ZS_TOP_K", "5"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.55"))
W_GENERAL  = float(os.getenv("W_GENERAL", "0.6"))
W_INDIAN   = float(os.getenv("W_INDIAN",  "0.6"))
W_CLIP     = float(os.getenv("W_CLIP",    "1.1"))
MAX_ITEMS  = int(os.getenv("MAX_ITEMS",   "4"))

# --------- Canonical dish names we want the model to land on ---------
CANDIDATES = [
  # Core US “health & diner” style
  "Oatmeal", "Porridge", "Yogurt Parfait", "Acai Bowl", "Smoothie Bowl",
  "Avocado Toast", "Grilled Cheese", "Chicken Wrap", "Beef Wrap", "Veggie Wrap",
  "Chicken & Rice", "Beef & Rice", "Rice (cooked)", "Chicken Breast", "Beef Steak",
  "Omelette", "Egg", "Scrambled Eggs",
  "Caesar Salad", "Greek Salad", "Garden Salad",
  "Pasta", "Spaghetti", "Mac and Cheese",
  "Burger", "Sandwich", "Turkey Sandwich", "Ham Sandwich", "Club Sandwich",
  "Taco", "Burrito", "Quesadilla",
  "Fries", "Soup", "Sushi", "Steak",
  "Pizza", "Pepperoni Pizza", "Cheese Pizza",
  "Bagel with Cream Cheese", "Peanut Butter Toast",
  "Granola with Milk", "Cereal with Milk",

  # Bowls / trendy
  "Chicken Bowl", "Beef Bowl", "Veggie Bowl",

  # Indian basics (we keep these too)
  "Biryani","Dosa","Idli","Poori","Paratha","Roti","Dal","Paneer","Samosa",
  "Masala Dosa","Chicken Curry","Mutton Curry","Fish Curry","Veg Curry",
  "Chole","Rajma","Upma","Vada","Pav Bhaji","Poha","Kheer","Gulab Jamun",
  "Curd (yogurt)",
]

# --------- Default grams (UI hint; your backend computes nutrition) ---------
DEFAULT_GRAMS: Dict[str,int] = {
  # US staples
  "Oatmeal":100, "Porridge":250, "Yogurt Parfait":200, "Acai Bowl":300, "Smoothie Bowl":300,
  "Avocado Toast":120, "Grilled Cheese":170,
  "Chicken Wrap":250, "Beef Wrap":260, "Veggie Wrap":240,
  "Chicken & Rice":100, "Beef & Rice":100,
  "Rice (cooked)":180, "Chicken Breast":150, "Beef Steak":180,
  "Omelette":140, "Egg":50, "Scrambled Eggs":150,
  "Caesar Salad":220, "Greek Salad":220, "Garden Salad":220,
  "Pasta":220, "Spaghetti":220, "Mac and Cheese":250,
  "Burger":180, "Sandwich":160, "Turkey Sandwich":170, "Ham Sandwich":170, "Club Sandwich":220,
  "Taco":120, "Burrito":250, "Quesadilla":220,
  "Fries":150, "Soup":300, "Sushi":200, "Steak":200,
  "Pizza":120, "Pepperoni Pizza":130, "Cheese Pizza":120,
  "Bagel with Cream Cheese":120, "Peanut Butter Toast":100,
  "Granola with Milk":200, "Cereal with Milk":200,
  "Chicken Bowl":100, "Beef Bowl":100, "Veggie Bowl":100,

  # Indian
  "Biryani":100,"Dosa":120,"Idli":50,"Poori":60,"Paratha":80,"Roti":50,"Dal":180,"Paneer":120,
  "Samosa":80,"Masala Dosa":140,"Chicken Curry":220,"Mutton Curry":220,"Fish Curry":220,
  "Veg Curry":220,"Chole":220,"Rajma":220,"Upma":200,"Vada":90,"Pav Bhaji":220,"Poha":160,
  "Kheer":150,"Gulab Jamun":70,"Curd (yogurt)":100,
}

# --------- Aliases / normalizers (messy labels → our canonical names) ---------
ALIASES: Dict[str,str] = {
  # Oatmeal / porridge variants
  "oatmeal":"Oatmeal","porridge":"Oatmeal","oats":"Oatmeal","oat porridge":"Oatmeal",
  "banana oatmeal":"Oatmeal","blueberry oatmeal":"Oatmeal",
  # Eggs & omelets
  "omelet":"Omelette","omelette":"Omelette","scrambled egg":"Scrambled Eggs","scrambled eggs":"Scrambled Eggs",
  "fried egg":"Egg","boiled egg":"Egg","egg":"Egg",
  # Toasts / sandwiches
  "avocado toast":"Avocado Toast","smashed avocado toast":"Avocado Toast",
  "grilled_cheese_sandwich":"Grilled Cheese","grilled cheese":"Grilled Cheese",
  "peanut butter toast":"Peanut Butter Toast","bagel with cream cheese":"Bagel with Cream Cheese",
  "turkey sandwich":"Turkey Sandwich","ham sandwich":"Ham Sandwich","club sandwich":"Club Sandwich",
  "sandwich":"Sandwich",
  # Bowls
  "acai bowl":"Acai Bowl","açaí bowl":"Acai Bowl","smoothie bowl":"Smoothie Bowl",
  "chicken bowl":"Chicken Bowl","beef bowl":"Beef Bowl","veggie bowl":"Veggie Bowl",
  # Wraps
  "chicken wrap":"Chicken Wrap","chicken caesar wrap":"Chicken Wrap","beef wrap":"Beef Wrap",
  "veggie wrap":"Veggie Wrap","turkey wrap":"Turkey Sandwich",
  # Rice combos
  "chicken rice":"Chicken & Rice","chicken and rice":"Chicken & Rice","rice and chicken":"Chicken & Rice",
  "beef rice":"Beef & Rice","beef and rice":"Beef & Rice","rice and beef":"Beef & Rice",
  "white rice":"Rice (cooked)","steamed rice":"Rice (cooked)","fried rice":"Rice (cooked)",
  # Meats
  "chicken breast":"Chicken Breast","grilled chicken":"Chicken Breast","butter chicken":"Chicken Curry",
  "steak":"Steak","beef steak":"Beef Steak",
  # Salads
  "caesar salad":"Caesar Salad","greek salad":"Greek Salad","garden salad":"Garden Salad","house salad":"Garden Salad",
  # Pasta
  "spaghetti":"Spaghetti","macaroni and cheese":"Mac and Cheese","mac & cheese":"Mac and Cheese",
  # Pizza
  "pepperoni pizza":"Pepperoni Pizza","cheese pizza":"Cheese Pizza","pizza":"Pizza",
  # Tex-Mex
  "taco":"Taco","burrito":"Burrito","quesadilla":"Quesadilla",
  # Breakfast cereals
  "granola with milk":"Granola with Milk","cereal with milk":"Cereal with Milk","cereal":"Cereal with Milk",
  # Indian mappings
  "curd":"Curd (yogurt)","yoghurt":"Curd (yogurt)","naan":"Roti","chapati":"Roti","phulka":"Roti",
  "chicken curry":"Chicken Curry","veg curry":"Veg Curry",
}

def _normalize(lbl: str) -> str:
  s = (lbl or "").strip().lower()
  if s in ALIASES: 
    return ALIASES[s]
  # fuzzy contains
  if "oat" in s or "porridge" in s: return "Oatmeal"
  if "avocado" in s and "toast" in s: return "Avocado Toast"
  if "grilled" in s and "cheese" in s: return "Grilled Cheese"
  if "wrap" in s and "chicken" in s: return "Chicken Wrap"
  if "wrap" in s and "beef" in s: return "Beef Wrap"
  if "wrap" in s and ("veg" in s or "veget" in s): return "Veggie Wrap"
  if "chicken" in s and "rice" in s: return "Chicken & Rice"
  if "beef" in s and "rice" in s: return "Beef & Rice"
  if "rice" in s: return "Rice (cooked)"
  if "chicken" in s and "curry" not in s: return "Chicken Breast"
  if "yogurt" in s and "parfait" in s: return "Yogurt Parfait"
  if "bowl" in s and "acai" in s: return "Acai Bowl"
  if "bowl" in s and "smoothie" in s: return "Smoothie Bowl"
  if "caesar" in s and "salad" in s: return "Caesar Salad"
  if "greek" in s and "salad" in s: return "Greek Salad"
  if "salad" in s: return "Garden Salad"
  if "spaghetti" in s: return "Spaghetti"
  if "mac" in s and "cheese" in s: return "Mac and Cheese"
  if "quesadilla" in s: return "Quesadilla"
  if "taco" in s: return "Taco"
  if "burrito" in s: return "Burrito"
  if "pizza" in s and "pepperoni" in s: return "Pepperoni Pizza"
  if "pizza" in s and "cheese" in s: return "Cheese Pizza"
  if "pizza" in s: return "Pizza"
  # title fallback (prefer candidate if exists)
  t = s.title()
  return t if t in CANDIDATES else t

class FoodEstimator:
  def __init__(self):
    self.cls_pipes = []
    self._try_add_cls(HF_MODEL_GENERAL)
    self._try_add_cls(HF_MODEL_INDIAN, optional=True)
    self.zs = None
    self._try_add_zs(HF_MODEL_ZS, optional=True)
    if not self.cls_pipes and not self.zs:
      raise RuntimeError("No vision model available")
    print("[estimator] preloaded model at startup")

  def _try_add_cls(self, model_id: str|None, optional=False):
    if not model_id: return
    try:
      self.cls_pipes.append(
        pipeline("image-classification", model=model_id,
                 trust_remote_code=True, use_auth_token=HF_TOKEN)
      )
      print(f"[FoodEstimator] Loaded {model_id}")
    except Exception as e:
      print(f"[FoodEstimator] {'Optional' if optional else 'Required'} model load failed: {model_id} -> {e}")

  def _try_add_zs(self, model_id: str|None, optional=False):
    if not model_id: return
    try:
      self.zs = pipeline("zero-shot-image-classification", model=model_id,
                         use_auth_token=HF_TOKEN)
      print(f"[FoodEstimator] Loaded zero-shot {model_id}")
    except Exception as e:
      if optional: print(f"[FoodEstimator] Optional zero-shot failed: {model_id} -> {e}")
      else:        print(f"[FoodEstimator] Zero-shot failed: {e}")

  def estimate(self, image_path: str|None=None, image_bytes: bytes|None=None) -> Dict[str,Any]:
    if not image_path and not image_bytes:
      raise ValueError("Provide image_path or image_bytes")
    img = Image.open(image_path) if image_path else Image.open(__import__("io").BytesIO(image_bytes))
    img = img.convert("RGB")

    # aggregate scores
    scores: Dict[str, float] = {}

    # 1) classifier heads
    if self.cls_pipes:
      pipe_weights = [W_GENERAL, W_INDIAN] + [0.6]*(max(0, len(self.cls_pipes)-2))
      for p, w in zip(self.cls_pipes, pipe_weights):
        try:
          preds = p(img, top_k=TOP_K)
          for pr in preds:
            raw = pr.get("label") or pr.get("class") or ""
            conf = float(pr.get("score", 0))
            norm = _normalize(raw)
            scores[norm] = max(scores.get(norm, 0.0), conf * w)
        except Exception as e:
          print("[FoodEstimator] cls error:", e)

    # 2) zero-shot re-rank
    if self.zs:
      try:
        zs = self.zs(img, candidate_labels=CANDIDATES, top_k=ZS_TOP_K)
        zs = [zs] if isinstance(zs, dict) else zs
        for pr in zs:
          norm = pr.get("label")
          conf = float(pr.get("score", 0))
          scores[norm] = max(scores.get(norm, 0.0), conf * W_CLIP)
      except Exception as e:
        print("[FoodEstimator] zero-shot error:", e)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    items = []
    for name, sc in ranked[:MAX_ITEMS]:
      items.append({
        "name": name,
        "default_grams": int(DEFAULT_GRAMS.get(name, 100)),
        "confidence": round(float(sc), 4)
      })

    if not items:
      items = [{"name":"Meal","default_grams":150,"confidence":0.0}]

    title = items[0]["name"]
    if items and items[0]["confidence"] < CONF_THRES and len(items) >= 2:
      title = f'{items[0]["name"]} or {items[1]["name"]}'

    return {
      "dish_name": title,
      "items": items,
      "preview_image_id": f"img_{uuid.uuid4().hex[:10]}",
    }