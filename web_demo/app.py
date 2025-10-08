from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
from datetime import datetime
import csv
from uuid import uuid4

app = Flask(__name__, instance_relative_config=True)
app.config.from_pyfile('config.py')

# Ensure app.secret_key exists (fallback) so Flask sessions work
if not app.secret_key:
    app.secret_key = os.environ.get('FLASK_SECRET') 

# Ensure each client gets a stable session id (persisted in cookie)
@app.before_request
def _ensure_session_id():
    if not session.get("_id"):
        session["_id"] = str(uuid4())
        session.modified = True

# Keep track of true catalog size (from recommender) but don't materialize all items for the UI.
CATALOG_MAX = None               # real size (filled after recommender loads)
UI_DISPLAY_MAX = 200             # max items to build for initial UI (page size)

# Load recommender
try:
    from utils.recommender import GRURecommender
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # web_demo folder
    MODEL_PATH = os.path.join(BASE_DIR, "model", "gru4rec_torch", "output_data", "save_model_test.pt")
    recommender = GRURecommender(MODEL_PATH, device="cpu")
    print("Recommender ok:", recommender.ok, "error:", recommender.error, "items:" if recommender.ok else "", len(recommender.itemidmap) if recommender.ok else "")
    # set catalog size immediately so /products returns total
    if recommender and getattr(recommender, "ok", False):
        try:
            CATALOG_MAX = len(recommender.itemidmap)
        except Exception:
            CATALOG_MAX = None
except Exception as e:
    recommender = None
    print("Recommender import exception:", e)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")
os.makedirs(DATA_DIR, exist_ok=True)

ENABLE_VIEW_LOGGING = False

def log_event(event_type, item_id=None, extra=None):
    if event_type == "view" and not ENABLE_VIEW_LOGGING:
        return
    if event_type.startswith("view_") and not ENABLE_VIEW_LOGGING:
        return
    try:
        with open(INTERACTIONS_PATH, "a", encoding="utf-8") as f:
            ts = datetime.utcnow().isoformat()
            sid = session.get("_id") or session.get("session_id") or ""
            hist_len = len(session.get("history", []))
            f.write(f"\n{ts},{sid},{event_type},{item_id or ''},{hist_len},{extra or ''}")
    except Exception as e:
        print("Log event failed:", e)

def _build_products(limit=None, offset=0):
    if not (recommender and recommender.ok):
        return []
    # determine catalog size lazily
    global CATALOG_MAX
    if CATALOG_MAX is None:
        try:
            CATALOG_MAX = len(recommender.itemidmap)
        except Exception:
            CATALOG_MAX = 0
    # only build a small page of product stubs for the UI
    limit = UI_DISPLAY_MAX if limit is None else limit
    # slice the index without converting entire index to list
    idx = recommender.itemidmap.index
    # pandas Index supports slicing; fallback to safe iteration
    try:
        slice_ids = idx[offset: offset + limit]
    except Exception:
        slice_ids = list(idx)[offset: offset + limit]
    return [{"id": str(it), "name": f"Item {it}", "quantity": 5} for it in slice_ids]

# build initial PRODUCTS page only (small)
PRODUCTS = _build_products(limit=UI_DISPLAY_MAX)
RECOMMENDATIONS = []

def get_history():
    return session.get("history", [])

def add_history(pid):
    pid = str(pid)
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

def _product_for_id(pid):
    """Return existing product or a lightweight stub for pid (pid may be int/str)."""
    pid = str(pid)
    p = next((p for p in PRODUCTS + RECOMMENDATIONS if p.get("id") == pid), None)
    if p:
        return p
    # Lightweight stub — you can enrich with images/prices later
    return {"id": pid, "name": f"Item {pid}", "quantity": 1}

@app.route("/")
def index():
    global PRODUCTS
    if (not PRODUCTS) and recommender and recommender.ok:
        PRODUCTS = _build_products()
    history = get_history()
    rec_ids = []
    sid = session.get("_id")
    if recommender and recommender.ok and history:
        # convert only numeric ids
        hist_int = []
        for x in history:
            try:
                hist_int.append(int(str(x).split("_", 1)[0]))
            except Exception:
                pass
        if hist_int:
            rec_ids = recommender.recommend(hist_int, topk=6, session_id=sid)
            rec_ids = [str(x) for x in rec_ids]
    # Do not fallback when history is empty; rec_ids will stay []
    recs = [_product_for_id(rid) for rid in rec_ids]      
    log_event("page_index", extra="|".join(history[-5:]))
    print(f"[INDEX] products={len(PRODUCTS)} history={history} rec_ids={rec_ids}")
    return render_template("index.html", products=PRODUCTS, recommendations=recs)

@app.route("/product/<product_id>")
def product_page(product_id):
    global RECOMMENDATIONS
    # try to find product among the small PRODUCTS page or cached RECOMMENDATIONS
    product = next((p for p in PRODUCTS + RECOMMENDATIONS if p["id"] == product_id), None)
    if not product:
        # create a lightweight stub so recommendation links work even when the full catalog
        # isn't materialized; cache it in RECOMMENDATIONS (bounded) so subsequent routes can find it
        product = _product_for_id(product_id)
        RECOMMENDATIONS.insert(0, product)
        # keep cache from growing unbounded
        if len(RECOMMENDATIONS) > 500:
            RECOMMENDATIONS.pop()
        log_event("view_product_stub", product_id)

    # record history and show product page
    add_history(product_id)
    if ENABLE_VIEW_LOGGING:
        log_event("view_product", product_id)

    # update recommender session and compute fresh recommendations
    sid = session.get("_id")
    if recommender and hasattr(recommender, "update_session"):
        try:
            recommender.update_session(sid, product_id)
        except Exception:
            pass
    history = get_history()
    rec_ids = []
    if recommender and recommender.ok and history:
        hist_int = []
        for x in history:
            try:
                hist_int.append(int(str(x).split("_", 1)[0]))
            except Exception:
                pass
        if hist_int:
            rec_ids = recommender.recommend(hist_int, topk=6, session_id=sid)
            rec_ids = [str(x) for x in rec_ids]
    recs = [_product_for_id(rid) for rid in rec_ids]

    return render_template("product.html", product=product, recommendations=recs)

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
    data = request.get_json(silent=True) or {}
    item_ids = data.get("item_ids")
    if item_ids is None:
        single = data.get("item_id")
        item_ids = [single] if single else []
    for iid in item_ids:
        s = str(iid)
        if s.endswith("_view"):
            if ENABLE_VIEW_LOGGING:
                log_event("view", s)
            continue  # skip view logs when disabled
        log_event("click", s)
    return "", 204

@app.route("/submit-checkout", methods=["POST"])
def submit_checkout():
    purchased_items = session.get("purchased", [])
    for item in purchased_items:
        log_event("complete_transaction", item.get("id"))
    session["purchased"] = []
    return redirect(url_for("index"))

def _load_clusters_csv(path):
    if not os.path.isfile(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    # dict: item_id -> cluster
    return dict(zip(df["item_id"].astype(str).values, df["cluster"].astype(int).values))

# On startup, try to load clusters file if present
BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # web_demo folder
CLUSTERS_CSV = os.path.join(BASE_DIR, 'model', 'gru4rec_torch', 'output_data', 'yoochoose_item_clusters.csv')
ITEM_CLUSTER_MAP = _load_clusters_csv(CLUSTERS_CSV)

@app.route("/clusters")
def clusters():
    # Return clusters for PRODUCTS only (grouped)
    grouped = {}
    for p in PRODUCTS:
        cid = ITEM_CLUSTER_MAP.get(str(p["id"]))
        if cid is None:
            continue
        grouped.setdefault(str(cid), []).append(p)
    return jsonify(grouped)

# Paginated products endpoint (e.g. /products?page=0&size=50)
@app.route("/products")
def products_api():
    try:
        page = max(int(request.args.get("page", "0")), 0)
        size = int(request.args.get("size", UI_DISPLAY_MAX))
        size = min(max(size, 1), 1000)  # clamp size
    except Exception:
        page, size = 0, UI_DISPLAY_MAX
    offset = page * size
    prods = _build_products(limit=size, offset=offset)
    total = len(recommender.itemidmap) if (recommender and getattr(recommender, "ok", False)) else (CATALOG_MAX or 0)
    return jsonify({"page": page, "size": size, "items": prods, "total": total})

@app.route("/admin/reload_model")
def admin_reload_model():
    global recommender, PRODUCTS, RECOMMENDATIONS, CATALOG_MAX, MODEL_PATH
    try:
        # Reload recommender (support both reload() or re-init)
        if recommender and hasattr(recommender, "reload"):
            recommender.reload(MODEL_PATH, device="cpu")
        else:
            from utils.recommender import GRURecommender
            recommender = GRURecommender(MODEL_PATH, device="cpu")

        ok = bool(recommender and getattr(recommender, "ok", False))
        CATALOG_MAX = len(recommender.itemidmap) if ok else 0
        PRODUCTS = _build_products(limit=UI_DISPLAY_MAX, offset=0) if ok else []
        RECOMMENDATIONS = []

        return jsonify({
            "ok": ok,
            "error": None if ok else getattr(recommender, "error", "unknown"),
            "catalog": CATALOG_MAX,
            "ui_items": len(PRODUCTS)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        sid = session.get("_id")
        # reset history trong Flask session
        session["history"] = []
        # reset hidden state trong recommender (nếu có)
        if recommender and hasattr(recommender, "reset_session"):
            recommender.reset_session(sid)
        return jsonify({"ok": True, "message": "History cleared"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})