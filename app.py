import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session,flash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.isotonic import IsotonicRegression
import joblib
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    explained_variance_score, 
    max_error, 
    median_absolute_error
)
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
import matplotlib.pyplot as plt
import io
import base64

# Set up the Flask application
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["MONGO_URI"] = "mongodb://localhost:27017/Regression.AI"
mongo=PyMongo(app)

# Dataset and preprocessing
DATASET_PATH = r'D:\datasets'
df = pd.read_csv(os.path.join(DATASET_PATH, 'powerconsumption.csv'))

df = df.iloc[:, 1:]  # This drops the first column
column_names = df.drop(columns=['PowerConsumption_Zone1']).columns.tolist()

# Drop unnecessary columns (PowerConsumption_Zone2, PowerConsumption_Zone3)
df = df.drop(columns=['PowerConsumption_Zone2', 'PowerConsumption_Zone3'])

# Features (X) and target (y)
X = df.drop(columns=['PowerConsumption_Zone1'])
y = df['PowerConsumption_Zone1']

# Scaling features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def get_model(model_name):

    if model_name == 'Isotonic Regression':
        # Special handling for Isotonic Regression
        def prepare_isotonic_data(X, y):
            # Create a combined sorted DataFrame using first feature
            df_combined = pd.DataFrame({'X': X[:, 0], 'y': y})
            df_sorted = df_combined.sort_values('X')
            
            # Convert to numpy arrays
            X_isotonic = df_sorted['X'].values.reshape(-1, 1)
            y_isotonic = df_sorted['y'].values
            
            return X_isotonic.ravel(), y_isotonic

        # Prepare data with the first feature
        X_isotonic, y_isotonic = prepare_isotonic_data(X_train, y_train)
        
        # Create and fit the Isotonic Regression model
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(X_isotonic, y_isotonic)
        return model
    
    elif model_name == 'Lasso':
        return Lasso(alpha=0.1)
    
    elif model_name == 'Quantile Regression':
        return Ridge(alpha=0.5)  # Placeholder for quantile regression
    
    elif model_name == 'Polynomial Regression':
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression().fit(X_poly, y_train)
        return model
    
    elif model_name == 'Linear Regression':
        return LinearRegression()
    
    elif model_name == 'Decision Tree':
        return DecisionTreeRegressor()
    
    elif model_name == 'Random Forest':
        return RandomForestRegressor()
    
    elif model_name == 'SVR':
        return SVR()
    
    elif model_name == 'KNN':
        return KNeighborsRegressor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user in MongoDB
        user = mongo.db.users.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            # Set session variables
            session['username'] = username
            session['user_id'] = str(user['_id'])
            
            flash('Login successful!', 'success')
            return redirect(url_for('choose'))
        else:
            flash('Invalid username or password!', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))

        existing_username = mongo.db.users.find_one({'username': username})
        if existing_username:
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))

        existing_email = mongo.db.users.find_one({'email': email})
        if existing_email:
            flash('Email already exists!', 'danger')
            return redirect(url_for('signup'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long!', 'danger')
            return redirect(url_for('signup'))

        if '@' not in email or '.' not in email:
            flash('Invalid email format!', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        user = {
            'first_name': first_name,
            'last_name': last_name,
            'username': username,
            'email': email,
            'password': hashed_password,
            'models_trained': [] 
        }

        mongo.db.users.insert_one(user)

        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/aboutUs')
def about_us():
    return render_template('aboutUs.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def generate_feature_graphs(username):
    plt.close('all')  # Close any existing plots
    
    # Remove existing graphs for this user
    mongo.db.feature_graphs.delete_many({'username': username})
    
    # Create graphs for each feature
    feature_graphs = []
    
    # Use the actual scaled data X_scaled
    for i, feature in enumerate(column_names):
        # Ensure we don't exceed the number of columns in X_scaled
        if i < X.shape[1]:
            plt.figure(figsize=(10, 6))
            plt.scatter(X[:, i], y, alpha=0.6)
            plt.title(f'{feature} vs Power Consumption (Zone 1)')
            plt.xlabel(feature)
            plt.ylabel('Power Consumption (Zone 1)')
            
            # Add regression line
            z = np.polyfit(X[:, i], y, 1)
            p = np.poly1d(z)
            plt.plot(X[:, i], p(X[:, i]), "r--", label='Trend Line')
            plt.legend()
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode the image
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Store in MongoDB
            graph_doc = {
                'username': username,
                'feature': feature,
                'image_base64': image_base64
            }
            mongo.db.feature_graphs.insert_one(graph_doc)
            
            feature_graphs.append({
                'feature': feature,
                'image_base64': image_base64
            })
    
    return feature_graphs

@app.route('/choose', methods=['GET'])
def choose():
    # Check if user is logged in
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    
    return render_template('choose.html')

@app.route('/explore_dataset', methods=['GET'])
def explore_dataset():
    # Check if user is logged in
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    
    # Generate and store feature graphs
    feature_graphs = generate_feature_graphs(session['username'])
    
    # Dataset summary statistics
    dataset_summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'features': column_names,  # Remove .tolist()
        'target': 'PowerConsumption_Zone1',
        'descriptive_stats': df.describe().to_dict()
    }
    
    return render_template('explore_dataset.html', 
                           feature_graphs=feature_graphs,
                           dataset_summary=dataset_summary)

@app.route('/models')
def models_page():
    # Display available models as buttons
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    return render_template('models.html')

@app.route('/train_model/<model_name>', methods=['POST'])
def train_model(model_name):
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    model = get_model(model_name)

    if model_name == 'Polynomial Regression':
        # Polynomial regression already fits on training data, no need to call fit here
        y_pred = model.predict(PolynomialFeatures(degree=2).fit_transform(X_test))
    
    elif model_name == 'Isotonic Regression':
        # Special handling for Isotonic Regression
        X_test_isotonic = X_test[:, 0].ravel()
        y_pred = model.predict(X_test_isotonic)
    
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate Regression Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mape = (abs((y_test - y_pred) / y_test) * 100).mean()
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    sse = np.sum((y_test - y_pred) ** 2)

    joblib.dump(model, f'{model_name}.pkl')
    
    mongo.db.users.update_one(
        {'username': session['username']},
        {'$push': {
            'models_trained': {
                'model_name': model_name,
                'mae': mae,
                'r2': r2,
            }
        }}
    )

    # Return metrics and results to the results page
    return render_template(
        'results.html',
        model_name=model_name,
        mae=mae, mse=mse, rmse=rmse, mape=mape, sse=sse,
        r2=r2, evs=evs, max_err=max_err, medae=medae
    )
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/justification')
def justification():
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    return render_template('justification.html')

@app.route('/train_justification/<model_name>', methods=['POST'])
def train_justification(model_name):
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    
    # Different train-test splits to try
    split_ratios = [0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9]
    results = []
    
    for test_size in [0.5,0.45, 0.4,0.35, 0.3,0.25, 0.2,0.15, 0.1]:  # Corresponds to train sizes of 50%, 60%, 70%, 80%, 90%
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model = get_model(model_name)
        
        if model_name == 'Polynomial Regression':
            poly = PolynomialFeatures(degree=2)
            X_train_poly = poly.fit_transform(X_train_split)
            X_test_poly = poly.transform(X_test_split)
            model.fit(X_train_poly, y_train_split)
            y_pred = model.predict(X_test_poly)
        elif model_name == 'Isotonic Regression':
            X_train_iso = X_train_split[:, 0].ravel()
            X_test_iso = X_test_split[:, 0].ravel()
            model.fit(X_train_iso, y_train_split)
            y_pred = model.predict(X_test_iso)
        else:
            model.fit(X_train_split, y_train_split)
            y_pred = model.predict(X_test_split)
        
        mae = mean_absolute_error(y_test_split, y_pred)
        mse = mean_squared_error(y_test_split, y_pred)
        r2 = r2_score(y_test_split, y_pred)
        
        train_size = round((1 - test_size) * 100)
        results.append({
            'train_size': train_size,
            'mae': mae,
            'mse': mse,
            'r2': r2
        })
    
    # Create visualization with bar plots
    plt.figure(figsize=(12, 6))
    train_sizes = [r['train_size'] for r in results]
    maes = [r['mae'] for r in results]
    r2s = [r['r2'] for r in results]
    
    # First subplot for MAE
    plt.subplot(1, 2, 1)
    bars_mae = plt.bar(train_sizes, maes, color='skyblue', alpha=0.7)
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE vs Training Size')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars for MAE
    for bar in bars_mae:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Second subplot for R²
    plt.subplot(1, 2, 2)
    bars_r2 = plt.bar(train_sizes, r2s, color='lightgreen', alpha=0.7)
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('R² Score')
    plt.title('R² vs Training Size')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars for R²
    for bar in bars_r2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Find best split ratio based on R² score
    best_result = max(results, key=lambda x: x['r2'])
    best_split = best_result['train_size']
    
    return render_template(
        'justification_results.html',
        model_name=model_name,
        results=results,
        plot_url=plot_url,
        best_split=best_split
    )
    
selected_features = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
X = df[selected_features]
y = df['PowerConsumption_Zone1']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

@app.route('/ensemble', methods=['GET'])
def ensemble():
    if 'username' not in session:
        flash('Please login to access this page', 'danger')
        return redirect(url_for('login'))
    
    return render_template('ensemble.html', features=selected_features)

@app.route('/ensemble_predict', methods=['POST'])
def ensemble_predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get input values for selected features
    input_data = []
    for feature in selected_features:
        input_data.append(float(request.form[feature]))
    
    input_scaled = scaler.transform([input_data])
    
    # Rest of the ensemble prediction code remains same...
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'Ridge': Ridge(alpha=0.5),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(input_scaled)[0]
        r2 = r2_score(y_test, model.predict(X_test))
        models[name] = (model, r2)
    
    total_r2 = sum(r2 for _, r2 in models.values())
    weights = {name: r2/total_r2 for name, (_, r2) in models.items()}
    final_prediction = sum(predictions[name] * weights[name] for name in models.keys())
    
    model_metrics = [
        {
            'name': name,
            'weight': f"{weights[name]:.3f}",
            'prediction': f"{predictions[name]:.2f}",
            'r2': f"{models[name][1]:.3f}"
        } for name in models.keys()
    ]
    
    return render_template('ensemble.html',
                         features=selected_features,
                         model_metrics=model_metrics,
                         prediction=f"{final_prediction:.2f}")
    
if __name__ == '__main__':
    app.run(debug=True)