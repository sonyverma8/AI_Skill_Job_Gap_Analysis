from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session, jsonify
import pandas as pd
import pickle
import re
import os
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SESSION_TYPE'] = 'filesystem'

MODEL_PATH = "skill_gap_ml_model.pkl"
DATA_PATH = "Skill_Job_Matching_Dataset.csv"
USER_PROFILES_PATH = "user_profiles.json"
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9,; ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

def train_model(csv_path=DATA_PATH):
    try:
        df = pd.read_csv(csv_path)
        cols = [c.lower() for c in df.columns]
        title_col = next((df.columns[i] for i, c in enumerate(cols) if c in ["job_title", "title", "jobtitle", "job title"]), None)
        skills_col = next((df.columns[i] for i, c in enumerate(cols) if c in ["skills", "skill_set", "required_skills", "requirements", "key_skills"]), None)
        desc_col = next((df.columns[i] for i, c in enumerate(cols) if "description" in c), None)
        exp_col = next((df.columns[i] for i, c in enumerate(cols) if "experience" in c), None)
        edu_col = next((df.columns[i] for i, c in enumerate(cols) if "education" in c), None)

        if not title_col:
            title_col = df.columns[0]

        feature_cols = []
        if skills_col: feature_cols.append(skills_col)
        if desc_col: feature_cols.append(desc_col)
        if exp_col: feature_cols.append(exp_col)
        if edu_col: feature_cols.append(edu_col)

        if not feature_cols:
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if title_col in text_cols:
                text_cols.remove(title_col)
            feature_cols = text_cols

        df["skills_text"] = df[feature_cols].astype(str).agg(" ".join, axis=1)
        df["skills_text"] = df["skills_text"].apply(clean_text)

        X = df["skills_text"]
        y = df[title_col].astype(str)

        vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words='english')
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Model trained with accuracy:", accuracy)
        feature_names = vectorizer.get_feature_names_out()
        job_classes = model.classes_
        feature_importance_per_class = {}
        
        for i, job_class in enumerate(job_classes):
            feature_importance = model.feature_importances_[i::len(job_classes)]
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]
            top_features = [(feature_names[idx], feature_importance[idx]) for idx in top_features_idx]
            feature_importance_per_class[job_class] = top_features

        model_bundle = {
            "vectorizer": vectorizer,
            "model": model,
            "title_col": title_col,
            "skills_col": skills_col,
            "accuracy": accuracy,
            "feature_importance": feature_importance_per_class
        }

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_bundle, f)

        return accuracy, feature_importance_per_class
    except Exception as e:
        print(f"Error training model: {e}")
        return 0, {}

def predict_job_and_gap(candidate_skills):
    bundle = load_model()
    if not bundle:
        return None, [], "Model not trained yet."

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    title_col = bundle["title_col"]
    skills_col = bundle.get("skills_col", None)

    skills_clean = clean_text(candidate_skills)
    X_vec = vectorizer.transform([skills_clean])
    predicted_job = model.predict(X_vec)[0]
    probabilities = model.predict_proba(X_vec)[0]
    job_classes = model.classes_
    top_jobs = sorted(zip(job_classes, probabilities), key=lambda x: x[1], reverse=True)[:5]

    missing_skills = []
    if skills_col:
        df = pd.read_csv(DATA_PATH)
        matched_jobs = df[df[title_col] == predicted_job]
        if not matched_jobs.empty:
            job_skills = " ".join(matched_jobs[skills_col].astype(str))
            job_skills_list = [s.strip().lower() for s in re.split(r'[;,]\s*', job_skills) if s.strip()]
            candidate_set = set([s.strip().lower() for s in re.split(r'[;,]\s*', candidate_skills)])
            missing_skills = list(set(job_skills_list) - candidate_set)

    return predicted_job, missing_skills, top_jobs, None

def get_job_recommendations(candidate_skills, top_n=5):
    bundle = load_model()
    if not bundle:
        return []
    
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    
    skills_clean = clean_text(candidate_skills)
    X_vec = vectorizer.transform([skills_clean])
    probabilities = model.predict_proba(X_vec)[0]
    job_classes = model.classes_
    top_jobs = sorted(zip(job_classes, probabilities), key=lambda x: x[1], reverse=True)[:top_n]
    df = pd.read_csv(DATA_PATH)
    title_col = bundle["title_col"]
    skills_col = bundle.get("skills_col", None)
    
    recommendations = []
    for job, probability in top_jobs:
        job_data = df[df[title_col] == job].iloc[0] if not df[df[title_col] == job].empty else None
        if job_data is not None:
            recommendations.append({
                'job_title': job,
                'probability': round(probability * 100, 2),
                'required_skills': job_data[skills_col] if skills_col and skills_col in job_data else "N/A"
            })
    
    return recommendations

def load_user_profiles():
    if os.path.exists(USER_PROFILES_PATH):
        with open(USER_PROFILES_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_user_profiles(profiles):
    with open(USER_PROFILES_PATH, 'w') as f:
        json.dump(profiles, f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            file.save(DATA_PATH)
            flash("Dataset uploaded successfully!", "success")
            return redirect(url_for("home"))
        else:
            flash("Please upload a valid CSV file.", "danger")
    return render_template("upload.html")

@app.route("/train", methods=["GET"])
def train():
    accuracy, feature_importance = train_model()
    return render_template("train.html", accuracy=accuracy, feature_importance=feature_importance)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    prediction = None
    gap = []
    top_jobs = []
    if request.method == "POST":
        skills = request.form["skills"]
        prediction, gap, top_jobs, error = predict_job_and_gap(skills)
        if error:
            flash(error, "danger")
        else:
            if 'user_id' in session:
                profiles = load_user_profiles()
                user_id = session['user_id']
                if user_id not in profiles:
                    profiles[user_id] = {'analyses': []}
                
                profiles[user_id]['analyses'].append({
                    'timestamp': datetime.now().isoformat(),
                    'skills': skills,
                    'prediction': prediction,
                    'gap': gap
                })
                
                save_user_profiles(profiles)
                
    return render_template("analyze.html", prediction=prediction, gap=gap, top_jobs=top_jobs)

@app.route("/dashboard")
def dashboard():
    bundle = load_model()
    accuracy = bundle["accuracy"] if bundle else None
    df = pd.read_csv(DATA_PATH)
    job_counts = df[df.columns[0]].value_counts().to_dict() if not df.empty else {}
    
    return render_template("dashboard.html", accuracy=accuracy, job_counts=job_counts)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/jobs")
def jobs():
    df = pd.read_csv(DATA_PATH)
    jobs_data = df.to_dict('records') if not df.empty else []
    return render_template("jobs.html", jobs=jobs_data)

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        user_id = request.form.get('user_id', str(datetime.now().timestamp()))
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        current_skills = request.form.get('skills', '')
        
        profiles = load_user_profiles()
        profiles[user_id] = {
            'name': name,
            'email': email,
            'skills': current_skills,
            'analyses': profiles.get(user_id, {}).get('analyses', [])
        }
        
        save_user_profiles(profiles)
        session['user_id'] = user_id
        flash("Profile saved successfully!", "success")
        return redirect(url_for('profile'))
    user_profile = None
    if 'user_id' in session:
        profiles = load_user_profiles()
        user_profile = profiles.get(session['user_id'], None)
    
    return render_template("profile.html", profile=user_profile)

@app.route("/recommendations")
def recommendations():
    skills = request.args.get('skills', '')
    if not skills and 'user_id' in session:
        profiles = load_user_profiles()
        user_profile = profiles.get(session['user_id'], {})
        skills = user_profile.get('skills', '')
    
    recommendations = get_job_recommendations(skills) if skills else []
    return render_template("recommendations.html", recommendations=recommendations, skills=skills)

@app.route("/api/job_skills/<job_title>")
def api_job_skills(job_title):
    bundle = load_model()
    if not bundle:
        return jsonify({"error": "Model not trained"})
    
    df = pd.read_csv(DATA_PATH)
    title_col = bundle["title_col"]
    skills_col = bundle.get("skills_col", None)
    
    if skills_col and skills_col in df.columns:
        job_data = df[df[title_col] == job_title]
        if not job_data.empty:
            skills = job_data[skills_col].iloc[0]
            return jsonify({"skills": skills})
    
    return jsonify({"error": "Job title not found"})

if __name__ == "__main__":
    app.run(debug=True)