import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    #With this the page title and layout are set
    page_title="Here should be our app title",
    layout="wide"
)


Properties = [
    #Here are the property data in a list of dictionaries
    {"name":"Property 1",
     "title":"Stylish old-building apartment",
     "locations_name":"Kreis 4",
     "lat": 47.3776,
     "lon": 8.5268,
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
     "lat": 47.4245,
     "lon": 9.3660,
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
     "title":"Family friendly townhouse",
     "locations_name":"Oerlikon",
     "lat": 47.4144,
     "lon": 8.5281,
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
     "title":"Historical Townhous in the heart of the city",
     "locations_name":"St.Gallen, Museumsquartier",
     "lat": 47.4274,
     "lon": 9.38160,
     "images": ["images/Prop4_A1.png"
                ,"images/Prop4_I1.png"
                ,"images/Prop4_I2.png"],
        "facts":{
            "price": "Fr. 1'850'000",
            "size": "160 sqm",
            "rooms": 6,
            "Building Year": 1900,
            "min_investment": "Fr. 85'000"
        },
        "pdf_factsheet_property": "factsheet/Factsheet4.pdf"
    }
]   

st.title("Real Estate Properties Overview") 
st.write("Please choose a property to see more details.")


st.header("Locations of the Properties")
# create the map and show were the properties are located 
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
st.map(map_data, zoom=7)


#List of properties with selectbox to choose from

for prop in Properties: 
    #each property will be displayed in a separate section becasue of the iteration
    with st.container (border=True):
    #here should the border be visible
    #and we create two columns, one for the image and one for the info
        spalte_Bild, spalte_Info = st.columns([1, 2])

    #Image column, Info column
        with spalte_Bild:
            st.image(prop["images"][0], use_container_width=True) 
            # shows the first image in the image list of the property
        with spalte_Info:
            # shows the property information while iterating through the properties
            st.subheader(prop["title"])
            st.caption(prop["locations_name"])
            st.text(f"Price: {prop['facts']['price']}")
            st.text(f"Minimum Investment: {prop['facts']['min_investment']}")

    #Button to show more details about the property/Expandable section
    
        with st.expander("Show more details"):

            # Show the image gallery
            st.subheader("Image Gallery")
            # This creates the tabs for the images while
            image_tabs = st.tabs([f"Image {i+1}" for i in range(len(prop["images"]))])
            for tab, img_url in zip(image_tabs, prop["images"]):
                with tab:
                    st.image(img_url, use_container_width=True)

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
            
            st.divider()

            # Diagram showing investment growth compared to XX over time (dummy data for now)
            st.subheader("Investment Growth compared to XXX")
            chart_data = pd.DataFrame(
                np.random.randn(20, 2),
                columns=["Your Investment", "XXX"]
            )
            st.line_chart(chart_data)

        

            st.divider()

             #pdf factsheet link download button
            st.link_button(
                "Download PDF Factsheet",
                prop["pdf_factsheet_property"], 
                use_container_width=True
                )

     


