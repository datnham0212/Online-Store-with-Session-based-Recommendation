import os
from flask import Flask, render_template, request, session, redirect, url_for
from flask import jsonify  # for JS interactions

app = Flask(__name__, instance_relative_config=True)
app.config.from_pyfile('config.py') # required for using session, here we use a random 24-byte key


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

def get_cart():
    return session.get("cart", [])

def add_to_cart(product_id):
    cart = session.get("cart", [])
    cart.append(product_id)
    session["cart"] = cart

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

@app.route("/add_to_cart/<product_id>")
def add_to_cart_route(product_id):
    add_to_cart(product_id)
    return redirect(url_for("cart"))

@app.route("/cart")
def cart():
    cart_ids = get_cart()
    cart_items = [p for p in PRODUCTS + RECOMMENDATIONS if p["id"] in cart_ids]
    return render_template("cart.html", cart_items=cart_items)

@app.route("/checkout")
def checkout():
    cart_ids = get_cart()
    cart_items = [p for p in PRODUCTS + RECOMMENDATIONS if p["id"] in cart_ids]
    return render_template("checkout.html", cart_items=cart_items)


