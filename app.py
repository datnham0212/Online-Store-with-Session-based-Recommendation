import os
from flask import Flask, render_template, request, session, redirect, url_for
from flask import jsonify  # for JS interactions

app = Flask(__name__, instance_relative_config=True)
app.config.from_pyfile('config.py') # required for using session, here we loaded from config

# Sample products (could be loaded from CSV/DB later)
PRODUCTS = [
    {"id": "a", "name": "Product A", "quantity": 5},
    {"id": "b", "name": "Product B", "quantity": 5},
    {"id": "c", "name": "Product C", "quantity": 5},
    {"id": "d", "name": "Product D", "quantity": 5},
    {"id": "e", "name": "Product E", "quantity": 5},
    {"id": "f", "name": "Product F", "quantity": 5},
]

RECOMMENDATIONS = []

def get_cart():
    return session.get("cart", {})

def get_cart_items_with_quantity():
    cart = get_cart()
    cart_items = []
    for pid, qty in cart.items():
        product = next((p for p in PRODUCTS + RECOMMENDATIONS if p["id"] == pid), None)
        if product:
            product_copy = product.copy()
            product_copy["quantity_in_cart"] = qty
            cart_items.append(product_copy)
    return cart_items

def add_to_cart(product_id):
    cart = session.get("cart", {})
    cart[product_id] = cart.get(product_id, 0) + 1
    session["cart"] = cart

def remove_one_from_cart(product_id):
    cart = session.get("cart", {})
    if product_id in cart:
        cart[product_id] -= 1
        if cart[product_id] <= 0:
            del cart[product_id]
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

@app.route("/remove_one_from_cart/<product_id>")
def remove_one_from_cart_route(product_id):
    remove_one_from_cart(product_id)
    return redirect(url_for("cart"))

@app.route("/remove_entire_from_cart/<product_id>")
def remove_entire_from_cart(product_id):
    cart = session.get("cart", {})
    if product_id in cart:
        del cart[product_id]
        session["cart"] = cart
    return redirect(url_for("cart"))

@app.route("/cart")
def cart():
    cart_items = get_cart_items_with_quantity()
    return render_template("cart.html", cart_items=cart_items)

@app.route("/purchase", methods=["POST"])
def purchase():
    cart_items = get_cart_items_with_quantity()
    session["purchased"] = cart_items  # Save a snapshot of cart at purchase time
    session["cart"] = {}  # Clear cart after purchase
    return redirect(url_for("checkout"))

@app.route("/checkout")
def checkout():
    purchased_items = session.get("purchased", [])
    return render_template("checkout.html", cart_items=purchased_items)



