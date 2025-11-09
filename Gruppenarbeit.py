import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.set_page_config(
    #With this the page title and layout are set
    page_title="Here should be our app title",
    layout="wide"
)



Properties = [
    #Here are the property data stored in a list of dictionaries
    {"name":"Property 1",
     "title":"Stylish 6 Apartment old-building in the city center",
     "locations_name":"Zurich, Aussershil",
     "lat": 47.3776,
     "lon": 8.5268,
     "description": "This stylish old-building apartment is located in the heart of Zurich's Aussershil district. Featuring six spacious apartments, this property combines classic architecture with modern amenities, offering a unique living experience in one of the city's most vibrant neighborhoods.",
     "images": ["images/Prop1_A1.png"
                ,"images/Prop1_I1.png"
                ,"images/Prop1_I2.png"],
        "facts": {
        "price": "Fr. 1'450'000",
        "size": "120 sqm",
        "rooms": 3,
        "Building Year": 1910,
        "min_investment": "Fr. 30'000"
        },
        "pdf_factsheet_property": "factsheet/Factsheet1.pdf"
    },


    {"name":"Property 2",
     "title":"Modern penthouse with city view",
     "locations_name":"St.Gallen Rosenberg",
     "lat": 47.427767,
     "lon": 9.363409,
        "description": "Contemporary penthouse located in the Rosenberg area of St.Gallen, offering stunning city views and modern living spaces. This property features an open floor plan, high-end finishes, and a private terrace, perfect for enjoying the vibrant urban lifestyle.",
     "images": ["images/Prop2_A1.png"
                ,"images/Prop2_I1.png"
                ,"images/Prop2_I2.png"],
        "facts": {
        "price": "Fr. 1'750'000",
        "size": "150 sqm",
        "rooms": 4,
        "Building Year": 2015,
        "min_investment": "Fr. 45'000"
        },
     "pdf_factsheet_property": "factsheet/Factsheet2.pdf"
    }, 
    {"name":"Property 3",
     "title":"Family Townhouse with big garden and BBQ area",
     "locations_name":"Oerlikon",
     "lat": 47.4144,
     "lon": 8.5281,
        "description": "Spacious family townhouse located in the Oerlikon district, featuring a large garden and BBQ area. This property offers ample living space, modern amenities, and a perfect setting for family gatherings and outdoor activities.",
     "images": ["images/Prop3_A1.png"
                ,"images/Prop3_I1.png"
                ,"images/Prop3_I2.png"],
        "facts": {
            "price": "Fr. 2'600'000",
            "size": "230 sqm",
            "rooms": 8,
            "Building Year": 2010,
            "min_investment": "Fr. 60'000"
        },
        "pdf_factsheet_property": "factsheet/Factsheet3.pdf"
    },
    {"name":"Property 4",
     "title":"Historical Townhouse in the heart of the city",
     "locations_name":"St.Gallen, Museumsquartier",
     "lat": 45.988696,
     "lon": 8.950601,
     "description": "Charming historical townhouse located in the Museumsquartier of St.Gallen. This beautifully preserved property features classic architecture, spacious interiors, and a rich history, offering a unique living experience in the heart of the city.",
     "images": ["images/Prop4_A1.png"
                ,"images/Prop4_I1.png"
                ,"images/Prop4_I2.png"],
        "facts":{
            "price": "Fr. 4'600'000",
            "size": "200 sqm",
            "rooms": 4.5,
            "Building Year": 2020,
            "min_investment": "Fr. 230'000"
        },
            "pdf_factsheet_property": "factsheet/Factsheet4.pdf"
        },
        {"name":"Property 5",
         "title":"Penthouse by the lake Lugano, a twist of amazing view and modern design",
         "locations_name":"Paradiso, Lugano",
         "lat": 47.4245,
         "lon": 9.3660,
         "description": "Luxury penthouse offering panoramic lake views, modern architecture, and premium amenities in Paradiso, Lugano. This exclusive residence features spacious rooms, private terrace, and top-tier finishes, ideal for sophisticated urban living.",
         "images": ["images/ChatGPT Image Nov 9, 2025 at 05_15_42 PM.png"
                    ,"images/ChatGPT Image Nov 9, 2025 at 05_15_44 PM.png"
                    ,"images/ChatGPT Image Nov 9, 2025 at 05_15_45 PM.png"],
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

st.title(":red[Real Estate Investment Platform]")
st.divider()

st.header("Real Estate Properties Overview") 
st.warning("Please choose a property to see more details.")



st.subheader("Locations of the Properties")
# create the map and show were the properties are locate 
# and find the lat and lon from the properties list(Dictionaries) 
# ith a list comprehension to iterate through the properties
map_data = pd.DataFrame(
    [{
        "lat": prop["lat"],
        "lon": prop["lon"],
        "name": prop["title"]
    } for prop in Properties]
)
#Zoom in so it is more clear where the properties are located
#A: here i added the pydeck library (which i found on streamlit) to create the map, so that we can change certain details of the map
# like the size of the dots and the color of the dots





map_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position='[lon, lat]',
    get_radius=2000,          # increase size to make dots bigger
    get_color=[255, 0, 0, 300], # RGB color with alpha, setting options for a red color with some transparency
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=46.8,   # center Switzerland-ish
    longitude=8.2,
    zoom=7,
    pitch=0,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[map_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}"}, # Show property name on hover
    )
)

#A:also with this change is possible to visualize the property names when hovering over the dots on the map
#List of properties with selectbox to choose from

for prop in Properties: 
    #each property will be displayed in a separate section because of the iteration
    with st.container (border=True):
    #here should the border be visible
    #and we create two columns, one for the image and one for the info
        spalte_Bild, spalte_Info = st.columns([1, 2])

#A: Changed the "use_contaier_width" to "use_column_width" in the st.image function to fix the bug
    
    #Image column, Info column
        with spalte_Bild:
            st.image(prop["images"][0], use_column_width=True)
            # shows the first image in the image list of the property
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
            for tab, img_url in zip(image_tabs, prop["images"]):
                with tab:
                    st.image(img_url, use_column_width=True)


            st.divider()

            # Divided so now we can have the slider/chart layout and the facts box side by side
            spalte_invest, spalte_facts = st.columns([2, 1]) # Main content 2/3, Facts box 1/3
            with spalte_invest:
                # Convert min_investment to integer for slider
                min_investment_int = int(prop["facts"]["min_investment"].replace("'", "").replace("Fr. ", ""))
                # Investment slider
                st.subheader("Your Investment")
                # Slider for investment amount, minimum investment and maximum 10,000,000 with step size of 1000
                investment_amount = st.slider("Select your investment amount:", min_investment_int, 10000000, min_investment_int, 1000, key=f"investement_slider_{prop['name']}")
                st.write(f"You selected: ${investment_amount}")

                st.subheader ("Time of Investment")
                # Slider for investment duration, minimum 1 year and maximum 30 years with step size
                investment_duration = st.slider("Select investment duration (years):", 1, 30, 10, 1, key=f"investment_duration_{prop['name']}")
                st.write(f"You selected: {investment_duration} years")

            
        
        

            with spalte_facts:
                # Facts box
                st.subheader("Property Facts")
                st.text(f"Price: {prop['facts']['price']}")
                st.text(f"Size: {prop['facts']['size']}")
                st.text(f"Rooms: {prop['facts']['rooms']}")
                st.text(f"Building Year: {prop['facts']['Building Year']}")
                st.text(f"Minimum Investment: {prop['facts']['min_investment']}")
                st.write("**Description**")
                st.write(prop.get("description", ""))
                st.divider()

            
            st.divider()

            # Diagram showing investment growth compared to XX over time (dummy data for now)
            st.subheader("Investment Growth compared to XXX")
            chart_data = pd.DataFrame(
                np.random.randn(20, 2),
                columns=["Your Investment", "XXX"]
            )
            st.line_chart(chart_data)

        
            
            st.divider()
            #open the pdf factsheet and create a download button
            import os

#A:Fixed the Bugs in the bugs in the code Below, now it allows the downloads effectively 

try:
    with open(prop["pdf_factsheet_property"], "rb") as f:
        st.download_button(
            label="Download Factsheet PDF",
            data=f,
            file_name=f"Factsheet_{prop['name']}.pdf",
            mime="application/pdf"
        )
except FileNotFoundError:
    st.info("Factsheet PDF not available.")


     


