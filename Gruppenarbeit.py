import os
import sqlite3

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

#i.a.: i searched for a data set with swiss properties (e.g. from github or kaggle) but they were too small or too synthetic!
#at first i chose a linear regression, but the metrics were too low (sth. with 25% r squared) so i chose the random forest model for better performance 
#i also generated histograms for the X and Y data to see how well they were distributed, therefore a log-transformation with the data seemed unnecessary as the data was well distributed 

#get data from github, link: https://github.com/1aaronh/ames_housing_prices/blob/master/data/ames.csv
v_df = pd.read_csv('ames.csv')


#creating a heat map to see which features correlate most with the price 
#plt.figure(figsize=(30, 18))
#sns.heatmap(v_df.select_dtypes(include=[np.number]).corr(), annot=True)
#plt.show()
#print(v_df)


#convert data into correct units 
v_df['Gr Liv Area'] = v_df['Gr Liv Area'] * 0.092903 #sqft to sqm
v_df['SalePrice'] = v_df['SalePrice'] * 0.80 #USD to CHF


#data cleaning --> price, area and number of rooms > 0
v_df = v_df[
    (v_df['SalePrice'] > 0) & 
    (v_df['TotRms AbvGrd'] > 0) & 
    (v_df['Gr Liv Area'] > 0)
]


#removing outliers using the iqr method 
#video: https://www.youtube.com/watch?v=9jYqZS142mg
def removing_outliers(v_df, column):
    Q1 = v_df[column].quantile(0.25)
    Q3 = v_df[column].quantile(0.75)
    IQR = Q3 - Q1 #calculating the interquartile (iqr) range (where the middle 50% of the values lie, the dispersion)
    lower_bound = Q1 - 1.5 * IQR #downard outliers
    upper_bound = Q3 + 1.5 * IQR #upward outliers 
    v_df_clean = v_df[(v_df[column] >= lower_bound) & (v_df[column] <= upper_bound)] #keeping the values between the limits, everything else is removed 
    return v_df_clean 


#removing rows with NaN in the used features 
features_needed = ['Year Built', 'TotRms AbvGrd', 'Gr Liv Area', 'SalePrice']
v_df_clean = v_df[features_needed].dropna()


#removing outliers (not with built year and number of rooms though)
v_df_clean = removing_outliers(v_df, 'SalePrice')
v_df_clean = removing_outliers(v_df, 'Gr Liv Area')


#defining the features X and target Y
X = v_df_clean[['Year Built', 'TotRms AbvGrd', 'Gr Liv Area']]
Y = v_df_clean['SalePrice']


#train/split test with 0.8 train and 0.2 for test, random state = 12 for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=12)


#random forest model 
crowdfunding = RandomForestRegressor(
    n_estimators=200, #200 "trees" are trained 
    max_depth=20, #chose depth 20 so that the model does not over- or underfit
    min_samples_split=5, #the "trees" should not be too complicated 
    min_samples_leaf=2,
    max_features='sqrt', 
    random_state=12, #same random state for reproducibility
)

#fitting the model 
crowdfunding.fit(X_train, Y_train)


#prediction
Y_pred = crowdfunding.predict(X_test)



#evaluating the model performance with r2, rmse and mae
#r2 = r2_score(Y_test, Y_pred) #measuring how well the features explain the variance, from 0-1
#rmse = np.sqrt(mean_squared_error(Y_test, Y_pred)) #standard deviation of the error, measures the average size of the prediction errors, in CHF
#mae = mean_absolute_error(Y_test, Y_pred) #average absolute error, measures the average absolute difference between the prediciton and the actual value, in CHF


#print(f"\nR_2 Score: {r2:.4f}") --> 0.7430 pretty solid for only 3 features 
#print(f"RMSE: CHF {rmse:,.2f}") --> -/+ 30,849.39 CHF normal for property prices (3 features only: location, condition, renovations, etc. are missing in the model)
#print(f"MAE: CHF {mae:,.2f}") --> 21,141.15 CHF also pretty solid 





# A: base dir + helper to resolve paths so images always load correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(rel_path: str) -> str:
    """Return absolute path for a file inside the project."""
    return os.path.join(BASE_DIR, rel_path)


# A: small debug – list images folder if it exists
images_dir = resolve_path("images")
if os.path.isdir(images_dir):
    print("DEBUG — images Ordner Inhalt:", os.listdir(images_dir))
else:
    print("DEBUG — images Ordner nicht gefunden unter:", images_dir)


st.set_page_config(
    #With this the page title and layout are set
    page_title="Crowdfunding platform",
    layout="wide"
)

#navigate to the Logo
logo_path = resolve_path("images/crowdl_logo.png")
#make an if else statement so it won't crash if it does not find the logo
if os.path.exists(logo_path):
    st.logo(logo_path, size="large")

else:
    print("Logo nicht gefunden")

#A: The next section is relevant for the Calculation section of the app, here Ivan created a whole lot of functions to cope with python shortcomings regarding number formatting and input validation

# The following helper functions are used to format and validate user inputs and outputs related to money, percentages, and years.

def format_thousands_ch(num: float, decimals: int = 0) -> str:
    s = f"{num:,.{decimals}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", "’")
    if decimals == 0:
        s = s.split(",")[0]
    return s


def show_money(n: float | None) -> str:
    if n is None:
        return ""
    return f"{format_thousands_ch(n, 0)} CHF"


def show_percent(ratio: float | None) -> str:
    if ratio is None:
        return ""
    return f"{ratio*100:.2f}%"


def show_years(y: float | None) -> str:
    if y is None:
        return ""
    return f"{y:.2f} years"


def strip_common_bits(text: str) -> str:
    s = str(text or "").strip()
    s = s.replace(" ", "").replace("’", "").replace("'", "")
    s = s.replace("CHF", "").replace("chf", "")
    s = s.replace("YEARS", "").replace("years", "").replace("Year", "").replace("year", "")
    s = s.replace("%", "")
    return s


def check_number_input(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a number."
    s = strip_common_bits(text)
    s = s.replace(",", "")
    try:
        return float(s), None
    except ValueError:
        return None, f"{label}: must be a valid number (e.g., 3’700’000 or 3700000)."


def check_years_input(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a positive number."
    s = strip_common_bits(text)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None, f"{label}: must be a valid number (e.g., 1.50)."
    if val <= 0:
        return None, f"{label}: must be greater than 0."
    return val, None


def check_percent_input(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a percentage."
    s = strip_common_bits(text)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    s = s.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None, f"{label}: must be a valid percentage (e.g., 6.25 or 6.25%)."
    ratio = val if val <= 1 else val / 100.0
    if ratio < 0:
        return None, f"{label}: cannot be negative."
    if ratio > 1:
        return None, f"{label}: cannot exceed 100%."
    return ratio, None


def check_crowdfunder_count(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a whole number from 1 to 19."
    s = strip_common_bits(text)
    if "." in s or "," in s:
        return None, f"{label}: must be an integer (no decimals)."
    if not s.isdigit():
        return None, f"{label}: must be an integer (e.g., 7)."
    val = int(s)
    if not (1 <= val <= 19):
        return None, f"{label}: must be between 1 and 19."
    return val, None


def touchup_money_key(key: str):
    raw = st.session_state.get(key, "")
    if raw is None or str(raw).strip() == "":
        return
    val, err = check_number_input("", raw)
    if err is None:
        st.session_state[key] = show_money(val)


def touchup_percent_key(key: str):
    raw = st.session_state.get(key, "")
    if raw is None or str(raw).strip() == "":
        return
    ratio, err = check_percent_input("", raw)
    if err is None:
        st.session_state[key] = show_percent(ratio)


def touchup_years_key(key: str):
    raw = st.session_state.get(key, "")
    if raw is None or str(raw).strip() == "":
        return
    years_val, err = check_years_input("", raw)
    if err is None:
        st.session_state[key] = show_years(years_val)


def show_readonly(label: str, value: str):
    if value:
        st.text_input(label, value=value, disabled=True)


def show_highlight_green(label: str, value: str):
    if value:
        st.success(f"{label}: {value}")


#THE OVERALL CODE STRUCTURE IS THE FOLLOWING:
#first we have the properties section with every property data stored in a list of dictionaries, and the different functionalities are built below (map. etc)
#second we have the CALCULATION SECTION, for the investment calculations


# A: DB init – create / refresh investment_returns.db with sample data

def reset_returns_table():
    db_path = resolve_path("investment_returns.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS returns;")
    cur.execute(
        """
        CREATE TABLE returns (
            year    INTEGER PRIMARY KEY,
            stocks  REAL,
            bonds   REAL,
            savings REAL,
            re_fund REAL
        );
        """
    )

    rows = [
        (2019, 0.3099,  0.020,  0.010,  0.150),
        (2020, 0.1994,  0.015,  0.008, -0.020),
        (2021, 0.2993,  0.010,  0.006,  0.100),
        (2022, -0.1876, -0.050, 0.015, -0.080),
        (2023, 0.2762,  0.020,  0.035,  0.060),
        (2024, 0.2448,  0.018,  0.040,  0.050),
    ]

    cur.executemany(
        """
        INSERT INTO returns (year, stocks, bonds, savings, re_fund)
        VALUES (?, ?, ?, ?, ?);
        """,
        rows
    )

    conn.commit()
    conn.close()


reset_returns_table()  # A: call once at startup so DB is always there


# This function loads investment data from SQLite-DB
def load_returns():
    db_path = resolve_path("investment_returns.db")

    try:
        #making connection to database
        conn = sqlite3.connect(db_path)

        query = """
        SELECT
            year,
            stocks,
            bonds,
            savings,
            re_fund
        FROM returns
        ORDER BY year;
        """

        df = pd.read_sql_query(query, conn)

        conn.close()

        if df.empty:
            return None

        return df

    except Exception as e:
        st.warning(f"Error loading data from database: {e}")
        return None


# A: HERE IS THE CODE SECTION OF THE PROPERTIES DATA

Properties = [
    #Here are the property data stored in a list of dictionaries
    {"id": 1,
     "name": "Property 1",
     "title": "Stylish 6 Apartment old-building in the city center",
     "locations_name": "Zurich, Aussershil",
     "lat": 47.3776,
     "lon": 8.5268,
     "description": "This stylish old-building apartment is located in the heart of Zurich's Aussershil district. Featuring six spacious apartments, this property combines classic architecture with modern amenities, offering a unique living experience in one of the city's most vibrant neighborhoods.",
     "images": [
         "images/Prop1_A1.png",
         "images/Prop1_I1.png",
         "images/Prop1_I2.png"
     ],
     "facts": {
         "price": "Fr. 1'450'000",
         "size": "120 sqm",
         "rooms": 3,
         "Building Year": 1910,
         "min_investment": "Fr. 30'000"
     },
     "pdf_factsheet_property": "factsheet/Factsheet1.pdf"
     },

    {"id": 2,
     "name": "Property 2",
     "title": "Modern penthouse with city and LAKE view",
     "locations_name": "Luzern, Tribschen",
     "lat": 47.0502,
     "lon": 8.3064,
     "description": "Contemporary penthouse located in the Tribschen area of Luzern, offering stunning city and lake views. This modern residence features open-concept living spaces, high-end finishes, and a private terrace, perfect for enjoying the picturesque surroundings and vibrant city life.",
     "images": [
         "images/Prop2_A1.png",
         "images/Prop2_I1.png",
         "images/Prop2_I2.png"
     ],

     "facts": {
         "price": "Fr. 1'750'000",
         "size": "150 sqm",
         "rooms": 4,
         "Building Year": 2015,
         "min_investment": "Fr. 45'000"
     },
     "pdf_factsheet_property": "factsheet/Factsheet2.pdf"
     },
    {"id": 3,
     "name": "Property 3",
     "title": "Family Townhouse with big garden and BBQ area",
     "locations_name": "Oerlikon",
     "lat": 47.4144,
     "lon": 8.5281,
     "description": "Spacious family townhouse located in the Oerlikon district, featuring a large garden and BBQ area. This property offers ample living space, modern amenities, and a perfect setting for family gatherings and outdoor activities.",
     "images": [
         "images/Prop3_A1.png",
         "images/Prop3_I1.png",
         "images/Prop3_I2.png"
     ],
     "facts": {
         "price": "Fr. 2'600'000",
         "size": "230 sqm",
         "rooms": 8,
         "Building Year": 2010,
         "min_investment": "Fr. 60'000"
     },
     "pdf_factsheet_property": "factsheet/Factsheet3.pdf"
     },
    {
        "id": 4,
        "name": "Property 4",
        "title": "Historical Townhouse in the heart of the city",
        "locations_name": "St.Gallen, Museumsquartier",
        "lat": 47.423821,
        "lon": 9.376152,
        "description": "Charming historical townhouse located in the Museumsquartier of St.Gallen. This beautifully preserved property features classic architecture, spacious interiors, and a rich history, offering a unique living experience in the heart of the city.",
        "images": [
            "images/Prop4_A1.png",
            "images/Prop4_I1.png",
            "images/Prop4_I2.png"
        ],
        "facts": {
            "price": "Fr. 4'600'000",
            "size": "200 sqm",
            "rooms": 4.5,
            "Building Year": 2020,
            "min_investment": "Fr. 230'000"
        },
        "pdf_factsheet_property": "factsheet/Factsheet4.pdf"
    },
    {"id": 5,
     "name": "Property 5",
     "title": "Penthouse by the lake Lugano, a twist of amazing view and modern design",
     "locations_name": "Paradiso, Lugano",
     "lat": 46.0037,
     "lon": 8.9556,
     "description": "Luxury penthouse offering panoramic lake views, modern architecture, and premium amenities in Paradiso, Lugano. This exclusive residence features spacious rooms, private terrace, and top-tier finishes, ideal for sophisticated urban living.",
     "images": ["images/ChatGPT Image Nov 9, 2025 at 05_15_42 PM.png",
                "images/ChatGPT Image Nov 9, 2025 at 05_15_44 PM.png",
                "images/ChatGPT Image Nov 9, 2025 at 05_15_45 PM.png"
                ],
     "facts": {
         "price": "Fr. 5'750'000",
         "size": "150 sqm",
         "rooms": 4,
         "Building Year": 2015,
         "min_investment": "Fr. 76'000"
     },
     "pdf_factsheet_property": "factsheet/Factsheet2.pdf"
     }
]

logo_path = resolve_path("images/crowdl_logo.png")

col_text, col_logo = st.columns([3,1])

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width="stretch")
    else:
        st.warning("Logo nicht gefunden")

#Help from AI to write a small html code so the title is centred, bigger and pushed down
with col_text:
    st.markdown("""
        <h1 style='text-align: center; color: #0F52BA; font-size: 6em; margin-bottom: 0; padding-top: 120px;'>
            Crowdle
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
        <h3 style='text-align: center; font-weight: normal; margin-top: 0;'>
            Gemeinsam in Schweizer Immobilien investieren
        </h3>
    """, unsafe_allow_html=True)
        
st.divider()
    

st.header("Real Estate Properties Overview")


st.subheader("Locations of the Properties")

map_data = pd.DataFrame(
    [{
        "lat": prop["lat"],
        "lon": prop["lon"],
        "name": prop["title"]
    } for prop in Properties]
)

#A: here i built a tooltip to show the property name and image when hovering over the dots on the map, so that a user can see more information about the property

BASE_URL = "https://raw.githubusercontent.com/jacometjoey-dotcom/Gruppenprojekt/main/"

rows = []
for prop in Properties:
    # Use first image for the tooltip preview
    img_path = prop["images"][0]
    raw_url = BASE_URL + img_path.replace(" ", "%20")

    rows.append({
        "lat": prop["lat"],
        "lon": prop["lon"],
        "name": prop["title"],
        "image_url": raw_url,
    })

map_data = pd.DataFrame(rows)

# simple scatter plot layer
map_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position='[lon, lat]',
    get_radius=2000,
    get_color=[255, 0, 0, 140],
    pickable=True,
)

#A: simple zoom at the beginning, nothing too fancy here

view_state = pdk.ViewState(
    latitude=46.8,
    longitude=8.5,
    zoom=6.5,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[map_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{name}</b><br><img src='{image_url}' width='180'>",
            "style": {"backgroundColor": "rgba(20,20,20,0.9)", "color": "white"}
        },
    )
)

#A:also with this change is possible to visualize the property names when hovering over the dots on the map
#List of properties with selectbox to choose from

for prop in Properties:
    #each property will be displayed in a separate section because of the iteration
    with st.container(border=True):
        #here should the border be visible
        #and we create two columns, one for the image and one for the info
        spalte_Bild, spalte_Info = st.columns([1, 2])

        #Image column, Info column
        with spalte_Bild:
            #A: use the GitHub raw URL also for the main picture
            img_rel = prop["images"][0]
            img_url = BASE_URL + img_rel.replace(" ", "%20")
            st.image(img_url, use_container_width=True)

        with spalte_Info:
            # shows the property information while iterating through the properties
            st.subheader(prop["title"])
            st.caption(prop["locations_name"])
            st.text(f"Price: {prop['facts']['price']}")
            st.text(f"Minimum Investment: {prop['facts']['min_investment']}")
            st.write(prop.get("description", ""))

        #Button to show more details about the property/Expandable section
        with st.expander("Show more details"):

            # Show the image gallery
            st.subheader("Image Gallery")
            # This creates the tabs for the images while
            image_tabs = st.tabs([f"Image {i+1}" for i in range(len(prop["images"]))])
            for tab, img_rel in zip(image_tabs, prop["images"]):
                with tab:
                    #A: again use GitHub raw URL for gallery images
                    img_url = BASE_URL + img_rel.replace(" ", "%20")
                    #created two collums so the pictures fit better 
                    col1_j,col2_j=st.columns([1,2]) 
                    with col1_j:
                        st.image(img_url,use_container_width=True)

            st.divider()

            # Divided so now we can have the slider/chart layout and the facts box side by side
            spalte_text_1, spalte_facts = st.columns([2, 1])  # Main content 2/3, Facts box 1/3


#title for the description 
            with spalte_text_1:
                st.subheader ("**Description**")
                #rendering the text in "blocksentence" (HTML essential for "justify")
                st.markdown(f"<div style='text-align: justify;'>{prop.get('description', '')}</div>", 
                    unsafe_allow_html=True)
#make a container with a seeable border 
            with spalte_facts:
                with st.container (border=True):
                    st.markdown("### Property Facts")
                    #make two columns
                    col_f1_j, col_f2_j = st.columns(2)

                    # br for the "Zeilenumbruch"
                    with col_f1_j:
                        st.markdown(f"**Price**<br>{prop['facts']['price']}", unsafe_allow_html=True)
                        st.markdown(f"**Size**<br>{prop['facts']['size']}", unsafe_allow_html=True)
                    with col_f2_j:
                        st.markdown(f"**Rooms**<br>{prop['facts']['rooms']}", unsafe_allow_html=True)
                        st.markdown(f"**Building Year**<br>{prop['facts']['Building Year']}", unsafe_allow_html=True)

                    st.divider()

                    #make the min investment green so it points more out
                    st.success(f"**Min. Investment:**\n\n### {prop['facts']['min_investment']}")


            st.divider()

            #open the pdf factsheet and create a download button

            pdf_path = resolve_path(prop["pdf_factsheet_property"])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Factsheet PDF",
                        data=f,
                        file_name=f"Factsheet_{prop['name']}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("Factsheet PDF not available.")


            sqm = int(prop["facts"]["size"].replace("sqm","").strip())
            rooms = prop["facts"]["rooms"]
            year_built = prop["facts"]["Building Year"]

            actual_price = int(prop["facts"]["price"]
                            .replace("fr","") #how is it in the data frame idk
                            .replace("Fr","")
                            .replace(".","")
                            .replace("'","")
                            .replace("",""))
            
            #predict price
            predicted_price = crowdfunding.predict([[sqm, rooms, year_built]])[0]

            #recommentation ->still need the percentage after (first like this simple)-> can delete ig
            if predicted_price < actual_price:
                recommendation = "Model recommends investing"
            else: 
                recommendation = "Model does not recommend investing"

#on the webiste (make it pretty green/red and text)
            st.subheader("Machine Learning Investment Recommendation")
            st.write(f"Predicted Price: {predicted_price:,.0f}".replace(",","'") + " CHF")
            st.write(f"Actual Price: {actual_price:,.0f}".replace(",","'") + " CHF")

            percentage_diff = ((predicted_price - actual_price) / actual_price) * 100
            if percentage_diff > 0:
                st.success(f"Machine Learning would recommend investing because the property is currently being sold for "
                          f"{abs(percentage_diff):.2f}% more than its estimated acquisition price.")
            else: 
                st.error(f"Machine Learning would NOT recommend investing because the property is currently being sold for "
                         f"{abs(percentage_diff):.2f}% less than its estimated acquisition price.")


            #Scatterplot: Living area vs sale price (Training data + selected property)

            st.subheader("Market Positioning: Living area vs. sale price")

            # here extracting the property features
            sqm = int(prop["facts"]["size"].replace("sqm","").strip())
            rooms = prop["facts"]["rooms"]
            year_built = prop["facts"]["Building Year"]

            # This is the clean actual price now 
            actual_price = int(
                prop["facts"]["price"]
                    .replace("Fr.","")
                    .replace("fr","")
                    .replace("'","")
                    .replace(" ","")
            )       

            # Creating the figure 
            fig, ax = plt.subplots(figsize=(10,6))

            # Plotting the training dataset 
            ax.scatter(
                v_df_clean["Gr Liv Area"],
                v_df_clean["SalePrice"],
                alpha=0.35,
                label="Training dataset"
            )

            # adding the plot for the selected property using a red dot
            ax.scatter(
                sqm,
                actual_price,
                color="red",
                s=140,
                label="Selected property"
            )

            # adding title and labels
            ax.set_title("Where does this property lie compared to the market?")
            ax.set_xlabel("Living area (sqm)")
            ax.set_ylabel("Sale price (CHF)")
            ax.legend()

            # running it on Streamlit
            st.pyplot(fig)




# A: HERE WE HAVE THE CALCULATION SECTION FOR THE INVESTMENT CALCULATIONS

#A:The following function implements the project calculator UI and logic.

def show_project_calculator():
    st.divider()
    st.header("Project Calculator (Developer View)")
    st.caption("Enter your project’s parameters and click **Calculate** to compute results.")

    #UI input fields
    st.subheader("Identification")
    st.text_input("Type of object", key="obj_type", placeholder="e.g., Multi-family house")
    st.text_input("Property address", key="addr", placeholder="Street, ZIP City")
    st.text_input("Property number", key="prop_no", placeholder="Parcel / registry no.")

    st.divider()
    st.subheader("Financial inputs")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input(
            "Purchase price of property",
            key="purchase_price_txt",
            placeholder="3’700’000 CHF",
            on_change=touchup_money_key,
            args=("purchase_price_txt",),
        )
        st.text_input(
            "Construction / renovation costs",
            key="renovation_costs_txt",
            placeholder="1’200’000 CHF",
            on_change=touchup_money_key,
            args=("renovation_costs_txt",),
        )
        st.text_input(
            "Financing ratio (bank)",
            key="bank_ratio_txt",
            placeholder="70.00%",
            on_change=touchup_percent_key,
            args=("bank_ratio_txt",),
        )
        st.text_input(
            "Mortgage interest rate",
            key="mortgage_rate_txt",
            placeholder="2.00%",
            on_change=touchup_percent_key,
            args=("mortgage_rate_txt",),
        )
    with c2:
        st.text_input(
            "Project duration (years)",
            key="years_txt",
            placeholder="1.50 years",
            on_change=touchup_years_key,
            args=("years_txt",),
        )
        st.text_input("Number of crowdfunders", key="n_crowdfunders_txt", placeholder="e.g., 10 (1–19)")
        st.text_input(
            "Profit participation of crowdfunders",
            key="profit_part_txt",
            placeholder="10.00%",
            on_change=touchup_percent_key,
            args=("profit_part_txt",),
        )
        st.text_input(
            "Interest paid to crowdfunders",
            key="crowd_rate_txt",
            placeholder="5.50%",
            on_change=touchup_percent_key,
            args=("crowd_rate_txt",),
        )

    st.text_input(
        "Developer equity (own funds)",
        key="own_equity_txt",
        placeholder="500’000 CHF",
        on_change=touchup_money_key,
        args=("own_equity_txt",),
    )
    st.text_input(
        "Sale price of property",
        key="sale_price_txt",
        placeholder="8’500’000 CHF",
        on_change=touchup_money_key,
        args=("sale_price_txt",),
    )

    # UI and calculations

    if st.button("Calculate"):
        error_list = []

        purchase_price_input, e = check_number_input("Purchase price of property", st.session_state.get("purchase_price_txt")); error_list += [e] if e else []
        reno_cost_input, e = check_number_input("Construction / renovation costs", st.session_state.get("renovation_costs_txt")); error_list += [e] if e else []
        bank_share_percent, e = check_percent_input("Financing ratio (bank)", st.session_state.get("bank_ratio_txt")); error_list += [e] if e else []
        mortgage_interest, e = check_percent_input("Mortgage interest rate", st.session_state.get("mortgage_rate_txt")); error_list += [e] if e else []
        project_years, e = check_years_input("Project duration (years)", st.session_state.get("years_txt")); error_list += [e] if e else []
        crowdfunder_count, e = check_crowdfunder_count("Number of crowdfunders", st.session_state.get("n_crowdfunders_txt")); error_list += [e] if e else []
        profit_share_percent, e = check_percent_input("Profit participation of crowdfunders", st.session_state.get("profit_part_txt")); error_list += [e] if e else []
        crowdfunder_interest, e = check_percent_input("Interest paid to crowdfunders", st.session_state.get("crowd_rate_txt")); error_list += [e] if e else []
        dev_equity_input, e = check_number_input("Developer equity (own funds)", st.session_state.get("own_equity_txt")); error_list += [e] if e else []
        sale_price_input, e = check_number_input("Sale price of property", st.session_state.get("sale_price_txt")); error_list += [e] if e else []

        if not error_list:
            if bank_share_percent > 1.00:
                error_list.append("Financing ratio (bank): cannot exceed 100%.")
            if profit_share_percent > 1.00:
                error_list.append("Profit participation of crowdfunders: cannot exceed 100%.")
            if mortgage_interest > 1.00:
                error_list.append("Mortgage interest rate: cannot exceed 100%.")
            if crowdfunder_interest > 1.00:
                error_list.append("Interest paid to crowdfunders: cannot exceed 100%.")

        if error_list:
            for msg in error_list:
                st.error(msg)
            st.stop()

        # ---- Calculations (same as your friend’s file) ----
        D11 = purchase_price_input
        D12 = reno_cost_input
        C15 = bank_share_percent
        C16 = mortgage_interest
        C17 = project_years
        C41 = crowdfunder_count
        C42 = profit_share_percent
        C43 = crowdfunder_interest
        D40 = dev_equity_input
        D50 = sale_price_input

        D13 = D11 + D12
        D18 = ((D13 * C15) * C16) * C17
        D19 = D13 * C15 * 0.005 * 2
        D20 = D13 * C15 * 0.005 * 2
        D21 = (D11 * 0.0475 + D50 * 0.0475) * 0.05 * (2 / 3)
        D22 = (D11 * 0.0475 + D50 * 0.0475) * 0.05 * (2 / 3)
        D23 = D12 * 0.12
        D24 = D50 - sum([D13, D18, D19, D20, D21, D22, D23])

        D27 = D11 * 0.02
        D28 = D50 * 0.0175
        D29 = D50 * 0.015
        D30 = D24 - D27 - D28 - D29

        D44 = D13 * (1 - C15) - D40
        D45 = D44 / C41
        D33 = (D45 * C43) * C17
        D34 = (D30 * C42) / C41
        D35 = (D33 + D34) / D45 / C17
        D36 = D35 * C17
        D37 = (D33 + D34) * C41
        D47 = D30 - D37

        D51 = D47 / D40 / C17
        D52 = D30 / (D13 * (1 - C15)) / C17
        D53 = D51 - D52

        # ---- Outputs ----
        st.divider()
        st.subheader("Key totals & costs")
        c3, c4 = st.columns(2)
        with c3:
            show_readonly("Total investment costs", show_money(D13))
            show_readonly("Total mortgage costs", show_money(D18))
            show_readonly("Transfer of ownership", show_money(D19))
            show_readonly("Modification of land charge", show_money(D20))
        with c4:
            show_readonly("Accounting / trust services", show_money(D21))
            show_readonly("Administration / tenant management", show_money(D22))
            show_readonly("Construction management", show_money(D23))
            show_readonly("Project result 1", show_money(D24))

        st.divider()
        st.subheader("Fees & project result 2")
        c5, c6 = st.columns(2)
        with c5:
            show_readonly("Broker fee (purchase)", show_money(D27))
            show_readonly("Broker fee (sale)", show_money(D28))
            show_readonly("Management fee (project mgmt)", show_money(D29))
        with c6:
            show_readonly("Project result 2", show_money(D30))
            show_readonly("Total financing through lenders", show_money(D44))
            show_readonly("Share per lender", show_money(D45))

        st.divider()
        st.subheader("Crowdfunder returns")
        c7, c8 = st.columns(2)
        with c7:
            show_readonly("Interest per lender (term)", show_money(D33))
            show_readonly("Profit per lender (term)", show_money(D34))
            show_readonly("ROI (p.a.)", show_percent(D35))
        with c8:
            show_readonly("ROI (total term)", show_percent(D36))
            show_readonly("Total distribution to crowdfunders", show_money(D37))

        st.divider()
        st.subheader("Project results: after financing & ROE")
        c9, c10 = st.columns(2)
        with c9:
            show_readonly("Project result after financing costs", show_money(D47))
            show_readonly("ROE (p.a.)", show_percent(D51))
        with c10:
            show_readonly("ROE (p.a.) without crowdfunding", show_percent(D52))
            show_highlight_green("Impact on ROE due to crowdfunding", show_percent(D53))

    else:
        st.caption("Adjust inputs and press **Calculate** to compute results.")


# A: run the calculator
show_project_calculator()

