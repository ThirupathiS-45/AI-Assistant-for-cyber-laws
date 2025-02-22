import os
import re
import pandas as pd
import joblib
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline    
from fpdf import FPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("Please set your actual Gemini API key in the .env file.")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)
print("Gemini API configured successfully!")

# Flask App Initialization
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SECRET_KEY"] = "your_secret_key"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# User Registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html")


# User Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password.", "danger")
    
    return render_template("login.html")

# User Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# Ensure dataset and model paths exist
DATA_FILE = "data/Full_Indian_Cyber_Laws.csv"
MODEL_PATH = "models/cyber_laws_classifier.pkl"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset file '{DATA_FILE}' not found.")

# Load dataset
df = pd.read_csv(DATA_FILE)

# Assign column names
law_column = "Section"
desc_column = "Offense"
punish_column = "Punishment"
case_type_column = "Case Type"

# Combine text for model training
df["text"] = df[law_column].astype(str) + " - " + df[desc_column].astype(str)

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Train and save ML model if not already saved
if not os.path.exists(MODEL_PATH):
    X = df["clean_text"]
    y = df[law_column]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

# Load trained model
pipeline = joblib.load(MODEL_PATH)

# Function to generate legal procedure using Gemini API
def get_legal_procedure(law_section):
    try:
        prompt = f"What are the legal procedures for {law_section} under Indian Cyber Law?"
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        return response.text if response else "No procedure found."
    except Exception as e:
        return f"Error retrieving procedure: {str(e)}"

# Function to predict law and generate legal procedure
def predict_law(query):
    query = clean_text(query)
    predicted_section = pipeline.predict([query])[0]
    
    law_details = df[df[law_column] == predicted_section].iloc[0]

    procedure = get_legal_procedure(predicted_section)

    return {
        "Section": predicted_section,
        "Offense": law_details[desc_column],
        "Punishment": law_details[punish_column],
        "Case Type": law_details[case_type_column],
        "Procedure": procedure
    }

# Function to generate PDF report
def generate_pdf(report_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Cyber Law Report", ln=True, align="C")
    pdf.ln(10)

    for key, value in report_data.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")
        pdf.ln()

    pdf_path = "static/cyber_law_report.pdf"
    os.makedirs("static", exist_ok=True)
    pdf.output(pdf_path)
    return pdf_path

# Flask Routes
@app.route("/")
@login_required  # 🔐 Users must log in to access home
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required  # 🔐 Protect this route
def predict():
    data = request.get_json()
    user_query = data.get("query") if data else None

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    result = predict_law(user_query)
    pdf_path = generate_pdf(result)

    return jsonify({
        "message": "Prediction successful",
        "data": result,
        "pdf_url": pdf_path
    })

@app.route("/download_report")
@login_required  # 🔐 Protect this route
def download_report():
    pdf_path = "static/cyber_law_report.pdf"
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return jsonify({"error": "Report not found"})

# Ensure database tables are created
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
