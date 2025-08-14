from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
from datetime import datetime

app = Flask(__name__, instance_relative_config=True)
app.config.from_pyfile('config.py')

CATALOG_MAX = 6  # adjust as needed

# Load recommender
try:
    from utils.recommender import GRURecommender
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_PATH = os.path.join(BASE_DIR, 'gru4rec_torch', 'output_data', 'save_model_test.pt')
    recommender = GRURecommender(MODEL_PATH, device="cpu")
    print("Recommender ok:", recommender.ok, "error:", recommender.error, "items:" if recommender.ok else "", len(recommender.itemidmap) if recommender.ok else "")
except Exception as e:
    recommender = None
    print("Recommender import exception:", e)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def log_event(event_type, item_id=None, extra=None):
    try:
        with open(INTERACTIONS_PATH, "a", encoding="utf-8") as f:
            ts = datetime.utcnow().isoformat()
            sid = session.get("_id") or session.get("session_id") or ""
            hist_len = len(session.get("history", []))
            f.write(f"{ts},{sid},{event_type},{item_id or ''},{hist_len},{extra or ''}\n")
    except Exception as e:
        print("Log event failed:", e)

def _build_products():
    if not (recommender and recommender.ok):
        return []
    ids = list(recommender.itemidmap.index)[:CATALOG_MAX]
    return [{"id": str(it), "name": f"Item {it}", "quantity": 5} for it in ids]

PRODUCTS = _build_products()
RECOMMENDATIONS = []

def get_history():
    return session.get("history", [])

def add_history(pid):
    hist = session.get("history", [])
    hist.append(pid)
    session["history"] = hist
    log_event("history_add", pid)

def get_cart():
    return session.get("cart", {})

def get_cart_items_with_quantity():
    cart = get_cart()
    out = []
    for pid, qty in cart.items():
        product = next((p for p in PRODUCTS + RECOMMENDATIONS if p["id"] == pid), None)
        if product:
            pc = product.copy()
            pc["quantity_in_cart"] = qty
            pc["max_quantity"] = product["quantity"]
            out.append(pc)
    return out

def add_to_cart(product_id):
    cart = session.get("cart", {})
    cart[product_id] = cart.get(product_id, 0) + 1
    session["cart"] = cart

@app.route("/")
def index():
    global PRODUCTS
    if (not PRODUCTS) and recommender and recommender.ok:
        PRODUCTS = _build_products()
    history = get_history()
    rec_ids = []
    if recommender and recommender.ok:
        rec_ids = recommender.recommend(history, topk=6)
    recs = [p for p in PRODUCTS if p["id"] in set(rec_ids)]
    log_event("page_index", extra="|".join(history[-5:]))
    print(f"[INDEX] products={len(PRODUCTS)} history={history} rec_ids={rec_ids}")
    return render_template("index.html", products=PRODUCTS, recommendations=recs)

@app.route("/product/<product_id>")
def product_page(product_id):
    product = next((p for p in PRODUCTS + RECOMMENDATIONS if p["id"] == product_id), None)
    if product:
        add_history(product_id)
        log_event("view_product", product_id)
        return render_template("product.html", product=product)
    log_event("view_missing", product_id)
    return "<h2>Product not found</h2>", 404

@app.route("/add_to_cart/<product_id>")
def add_to_cart_route(product_id):
    add_to_cart(product_id)
    add_history(product_id)
    log_event("add_to_cart", product_id)
    return redirect(url_for("cart"))

@app.route("/remove_from_cart/<product_id>")
def remove_from_cart(product_id):
    cart = session.get("cart", {})
    if product_id in cart:
        del cart[product_id]
        session["cart"] = cart
    return redirect(url_for("cart"))

@app.route("/buy_now/<product_id>")
def buy_now(product_id):
    product = next((p for p in PRODUCTS + RECOMMENDATIONS if p["id"] == product_id), None)
    if product:
        purchased_item = product.copy()
        purchased_item["quantity_in_cart"] = 1
        session["purchased"] = [purchased_item]
    return redirect(url_for("checkout"))

@app.route("/cart")
def cart():
    return render_template("cart.html", cart_items=get_cart_items_with_quantity())

@app.route("/purchase", methods=["POST"])
def purchase():
    session["purchased"] = get_cart_items_with_quantity()
    for it in session["purchased"]:
        log_event("purchase_add", it["id"])
    session["cart"] = {}
    return redirect(url_for("checkout"))

@app.route("/checkout")
def checkout():
    log_event("page_checkout")
    return render_template("checkout.html", cart_items=session.get("purchased", []))

# --- Added endpoints for explicit interaction logging ---

@app.route("/log_click", methods=["POST"])
def log_click():
    # Expect JSON: { "item_ids": ["123","456"] } or { "item_id": "123" }
    data = request.get_json(silent=True) or {}
    item_ids = data.get("item_ids")
    if item_ids is None:
        single = data.get("item_id")
        item_ids = [single] if single else []
    for iid in item_ids:
        log_event("click", iid)
    return "", 204

@app.route("/submit-checkout", methods=["POST"])
def submit_checkout():
    purchased_items = session.get("purchased", [])
    for item in purchased_items:
        log_event("complete_transaction", item.get("id"))
    session["purchased"] = []
    return redirect(url_for("index"))

