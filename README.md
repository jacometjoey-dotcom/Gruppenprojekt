# Crowdle 🏠

Click the link to open the website in your browser

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gruppenprojekt-uy34qhoxjjxg5kfzxfxubw.streamlit.app/)

**Crowdle** is a Swiss real estate crowdfunding platform that enables users to explore and invest in properties across Switzerland. The platform combines interactive property browsing with machine learning-powered price predictions to help potential investors make informed decisions.

## 🎯 What We're Building

This project is a **Streamlit web application** that offers:

- **Property Marketplace**: Browse curated Swiss real estate investment opportunities with detailed information, images, and downloadable factsheets
- **ML-Powered Price Predictions**: Get property value estimates using a Random Forest Regressor model trained on housing data and adjusted for the Swiss market
- **Investment Calculator**: Calculate ROI, mortgage costs, and projected returns based on your investment parameters
- **Market Analysis**: Visualize how properties compare to market trends with interactive charts and geographical mapping
- **Investment Recommendations**: Receive data-driven investment suggestions based on machine learning analysis

### Key Features

1. **Interactive Property Browser** - Explore 5+ properties with galleries, descriptions, and key facts
2. **Smart Investment Analysis** - ML model predicts if a property is over/underpriced
3. **Financial Calculator** - Compute ROI, mortgage payments, and cash flow projections
4. **Geographic Visualization** - Interactive map showing property locations across Switzerland
5. **Market Positioning Charts** - Compare properties against training dataset

## 🚀 Quick Start

### Prerequisites

- **Python 3.12 or higher** (required)
  - ⚠️ **Important**: This application requires Python 3.12+. Earlier versions (3.11 and below) may cause TypeErrors due to type hinting compatibility issues.
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Gruppenprojekt-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Gruppenarbeit.py
   ```

4. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`
   
   If it doesn't open automatically, navigate to the URL shown in your terminal.

## 📁 Project Structure

```
Gruppenprojekt-main/
│
├── Gruppenarbeit.py           # Main Streamlit application (entry point)
├── propertydata.py            # Property data definitions and storage
├── definitions.py             # Helper functions (formatting, validation, ML utilities)
├── requirements.txt           # Python dependencies
│
├── ames.csv                   # Training dataset (Ames Housing, adjusted for Swiss market)
├── data.csv                   # Additional data file
│
├── Feature_01.py              # Feature module 1 (placeholder utilities)
├── Feature_02.py              # Feature module 2 (placeholder utilities)
│
├── factsheet/                 # Property factsheet PDFs
│   ├── Factsheet1.pdf
│   ├── Factsheet2.pdf
│   ├── Factsheet3.pdf
│   ├── Factsheet4.pdf
│   └── Factsheet5.pdf
│
├── images/                    # Property images and logos
│   ├── crowdl_logo.png
│   ├── Prop1_A1.png          # Property 1 images
│   ├── Prop1_I1.png
│   ├── Prop1_I2.png
│   ├── Prop2_*.png           # Property 2 images
│   ├── Prop3_*.png           # Property 3 images
│   └── Prop4_*.png           # Property 4 images
│
└── README.md                  # This file
```

### File Descriptions

#### Core Application Files

- **[`Gruppenarbeit.py`](Gruppenarbeit.py)** (558 lines)
  - Main application entry point
  - Implements Streamlit UI with multiple pages/sections
  - Contains ML model training and prediction logic
  - Handles property visualization and user interactions

- **[`propertydata.py`](propertydata.py)**
  - Centralized property data storage
  - Contains dictionary of all available properties with metadata
  - Each property includes: title, location, coordinates, images, facts, and factsheet path

- **[`definitions.py`](definitions.py)** (151 lines)
  - Helper functions for data manipulation
  - Input validation (money, percentages, years)
  - Swiss number formatting (e.g., `1'450'000 CHF`)
  - Outlier removal using IQR method
  - Streamlit session state management

#### Data Files

- **[`ames.csv`](ames.csv)**
  - Training data for the ML model
  - Originally from the Ames Housing dataset (Iowa, USA)
  - Converted to Swiss market using adjustment factors:
    - Location/Land Factor: 4.8x (accounts for Zurich vs. Iowa price differences)
    - Construction Quality Factor: 1.6x (solid vs. stick-built construction)

- **[`requirements.txt`](requirements.txt)**
  - Lists all Python package dependencies
  - Ensures reproducible environment across systems

## 🤖 Machine Learning Model

### Overview

The application uses a **Random Forest Regressor** to predict property prices based on:

- **Year Built** - Construction year of the property
- **Total Rooms** - Number of rooms above ground
- **Living Area** - Size in square meters (sqm)

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=200,      # 200 decision trees
    max_depth=20,          # Maximum depth to prevent overfitting
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples per leaf
    max_features='sqrt',   # Number of features to consider
    random_state=12        # For reproducibility
)
```

### Data Processing Pipeline

1. **Data Loading**: Import Ames Housing dataset
2. **Unit Conversion**: Convert sq.ft → sqm (×0.092903)
3. **Market Adjustment**: Apply Swiss market factors (×7.68 total)
4. **Data Cleaning**: Remove rows with missing values or invalid data
5. **Outlier Removal**: Use IQR method to filter outliers in price and living area
6. **Train/Test Split**: 80% training, 20% testing (random_state=12)
7. **Model Training**: Fit Random Forest on training data
8. **Prediction**: Generate price estimates for new properties

### Model Performance

The model provides reasonable predictions for Swiss properties by adjusting US housing data. Performance metrics are calculated during training but can be extended further.

## 💡 How to Use the Application (Browser)

### 1. Browse Properties

- Navigate to the **Properties** section
- View property details including:
  - Location and interactive map
  - Image galleries
  - Price, size, rooms, building year
  - Minimum investment required
  - Downloadable PDF factsheets

### 2. Get ML Investment Recommendation

- Select a property to view
- The ML model automatically predicts the property's "fair value"
- Compare predicted price vs. asking price
- Receive a recommendation:
  - ✅ **Invest**: Property is underpriced (good deal)
  - ❌ **Don't Invest**: Property is overpriced

### 3. Calculate Investment Returns

- Go to the **Calculator** section
- Input your parameters:
  - Purchase price
  - Renovation costs
  - Financing ratio (bank loan percentage)
  - Mortgage interest rate
  - Expected sale price
  - Time horizon
- View calculated metrics:
  - Total acquisition cost
  - Required equity
  - Annual mortgage cost
  - Gross return
  - Net ROI

### 4. Analyze Market Position

- View scatter plots comparing properties to the training dataset
- Understand where your selected property sits in the market
- Identify trends in living area vs. sale price

## 📊 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web framework for the interactive UI |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **scikit-learn** | Machine learning (Random Forest) |
| **Matplotlib** | Data visualization and plotting |
| **Seaborn** | Statistical data visualization |
| **PyDeck** | Interactive geographic mapping |

## 🐛 Troubleshooting

### Common Issues

**Problem**: App doesn't start
```bash
# Solution: Install all dependencies
pip install -r requirements.txt --upgrade
```

**Problem**: Images not loading
```bash
# Solution: Check that all image files exist in the images/ directory
# Verify paths in propertydata.py match actual filenames
```

**Problem**: Port already in use
```bash
# Solution: Run on a different port
streamlit run Gruppenarbeit.py --server.port 8502
```
