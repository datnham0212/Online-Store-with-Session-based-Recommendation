from flask import Flask, render_template, request

app = Flask(__name__)

# Sample products (could be loaded from CSV/DB later)
PRODUCTS = [
    {"id": "a", "name": "Product A"},
    {"id": "b", "name": "Product B"},
    {"id": "c", "name": "Product C"},
    {"id": "d", "name": "Product D"},
    {"id": "e", "name": "Product E"},
    {"id": "f", "name": "Product F"},
]

RECOMMENDATIONS = [
    {"id": "x", "name": "Product X"},
    {"id": "y", "name": "Product Y"},
    {"id": "z", "name": "Product Z"},
]

@app.route("/")
def index():
    return render_template("index.html", products=PRODUCTS, recommendations=RECOMMENDATIONS)


@app.route("/log_click", methods=["POST"])
def log_click():
    data = request.get_json()
    item_id = data.get("item_id")
    
    with open("data/interactions.csv", "a") as f:
        f.write(f"{item_id}\n")  # Extend to include session_id, timestamp if needed

    return "", 204  # No Content

@app.route("/product/<product_id>")
def product_page(product_id):
    product = next((p for p in PRODUCTS + RECOMMENDATIONS if p["id"] == product_id), None)
    if product:
        return render_template("product.html", product=product)
    return "<h2>Product not found</h2>", 404

