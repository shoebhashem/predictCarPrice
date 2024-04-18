from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
import joblib

def load_model():
    return joblib.load('saved_model.pkl')

data = load_model()


reg = data["rf_reg"]
#reg1 = data["model_2"]
le_make = data["make"]
le_model = data["model"]
le_color = data["color"]
le_interior = data["interior"]
#le_state = data["state"]
le_transmission = data["transmission"]
#scaler = data["scaler_1"]
#scaler1 = data["scaler_2"]

models_by_make = {
"Kia": ['Sorento', 'Optima', 'K900', 'Rio', 'Soul', 'Forte', 'Sportage', 'Sedona',
        'Spectra', 'Rondo', 'Borrego', 'Cadenza', 'Amanti', 'Sephia'],
"BMW": ['3 Series', '6 Series Gran Coupe', 'M5', '6 Series', '5 Series', 'X6', 'X1',
        '4 Series', 'X5', '7 Series', '1 Series', 'M3', 'X3', 'Z4', 'X5 M',
        '5 Series Gran Turismo', 'X6 M', 'Z3', 'M', 'M4', 'ActiveHybrid X6', 'M6',
        '2 Series', 'M6 Gran Coupe', '3 Series Gran Turismo', 'ActiveHybrid 7',
        'Z4 M', 'i8', 'ActiveHybrid 5', 'X4', '4 Series Gran Coupe'],
"Nissan": ['Altima', 'Versa', 'Versa Note', '370Z', 'Juke', 'NV', 'Leaf', 'NV200', 'Rogue',
        'Maxima', 'Xterra', 'Sentra', 'Frontier', 'Cube', 'Armada', 'Pathfinder',
        'Murano', 'GT-R', 'Altima Hybrid', '350Z', 'Titan', 'Quest', 'Rogue Select',
        'Truck', 'Murano CrossCabriolet', '200SX', 'NV Cargo', 'NV Passenger', '300ZX'],
"Chevrolet": ['Cruze', 'Camaro', 'Impala', 'Suburban', 'Malibu', 'Silverado 1500', 'Traverse',
                'Silverado 2500HD', 'Equinox', 'Captiva Sport', 'Volt', 'Express Cargo',
                'Colorado', 'Express', 'Sonic', 'HHR', 'Tahoe', 'Impala Limited', 'Aveo',
                'Corvette', 'malibu', 'Avalanche', 'Tahoe Hybrid', 'Malibu Classic', 'Cobalt',
                'Uplander', 'Silverado 1500 Classic', 'Monte Carlo', 'TrailBlazer',
                'Malibu Maxx', 'Blazer', 'Cavalier', 'Venture', 'S-10', 'Silverado 1500HD',
                'Spark', 'SSR', 'Silverado 3500HD', 'Silverado 1500 Hybrid',
                'Silverado 2500HD Classic', 'Silverado 3500', 'TrailBlazer EXT', 'Classic',
                'Silverado 2500', 'Tracker', 'Prizm', 'Astro', 'Tahoe Limited/Z71',
                'C/K 1500 Series', 'Lumina', 'Black Diamond Avalanche', 'Corvette Stingray',
                'SS', 'Astro Cargo', 'Malibu Hybrid', 'C/K 3500 Series', 'Caprice',
                'C/K 2500 Series', 'Silverado 3500 Classic', 'S-10 Blazer', 'Spark EV'],
"Ford": ['Fusion', 'Escape', 'Edge', 'Focus', 'F-350 Super Duty', 'Fiesta', 'Flex',
        'Explorer', 'F-150', 'E-Series Van', 'E-Series Wagon', 'Expedition', 'Mustang',
        'Transit Connect', 'Taurus', 'Escape Hybrid', 'Ranger', 'Fusion Hybrid',
        'F-250 Super Duty', 'Econoline Cargo', 'Econoline Wagon', 'F-450 Super Duty',
        'Explorer Sport Trac', 'Five Hundred', 'Freestyle', 'Freestar', 'Excursion',
        'Thunderbird', 'Contour', 'Escort', 'Mustang SVT Cobra', 'Shelby GT500',
        'C-Max Energi', 'C-Max Hybrid', 'Crown Victoria', 'Taurus X', 'Expedition EL',
        'Focus ST', 'Fusion Energi', 'F-150 Heritage', 'Explorer Sport', 'Windstar',
        'Tempo', 'Windstar Cargo', 'F-150 SVT Lightning', 'E-150', 'Aspire',
        'Transit Van', 'E-250', 'Transit Wagon'],
"Hyundai": ['Elantra', 'Santa Fe', 'Genesis', 'Equus', 'Sonata', 'Sonata Hybrid', 'Accent',
                'Veloster', 'Elantra Coupe', 'Azera', 'Tucson', 'Genesis Coupe', 'Veracruz',
                'Santa Fe Sport', 'Elantra GT', 'Elantra Touring', 'Tiburon', 'Entourage',
                'XG350', 'XG300'],
"Toyota": ['Corolla', 'Sienna', 'Yaris', 'Camry', 'Tacoma', 'FJ Cruiser', 'Avalon',
        'Tundra', 'Prius', 'Camry Hybrid', '4Runner', 'Sequoia', 'Venza', 'Highlander',
        'Highlander Hybrid', 'RAV4', 'Prius Plug-in', 'Land Cruiser', 'Camry Solara',
        'Celica', 'Matrix', 'Prius v', 'Tercel', 'Prius c', 'ECHO', 'MR2 Spyder',
        'Pickup', 'Avalon Hybrid', 'T100'],
"Dodge": ['Avenger', 'Journey', 'Charger', 'Grand Caravan', 'Nitro', 'Challenger', 'Dart',
        'Caliber', 'Magnum', 'Durango', 'Ram Pickup 1500', 'Sprinter Cargo', 'Dakota',
        'Neon', 'Stratus', 'Ram Pickup 3500', 'Intrepid', 'Ram Pickup 2500', 'Caravan',
        'Ram Cargo', 'Viper', 'Sprinter'],
"Chrysler": ['200', '300', 'Town and Country', 'Sebring', 'PT Cruiser', 'Pacifica',
                'Concorde', 'LHS', 'Aspen', 'Crossfire', '300M', 'Prowler', 'Cirrus', 'Voyager'],
"Honda": ['Accord', 'CR-V', 'Civic', 'Fit', 'Pilot', 'Odyssey', 'Crosstour', 'CR-Z',
        'Accord Crosstour', 'Insight', 'Ridgeline', 'S2000', 'Element', 'accord',
        'Passport', 'Prelude', 'Accord Hybrid']
}

color_options = [
        "BLACK",
        "WHITE",
        "SILVER",
        "GRAY",
        "BLUE",
        "RED",
        "—",
        "GOLD",
        "GREEN",
        "BURGUNDY",
        "BEIGE",
        "BROWN",
        "ORANGE",
        "PURPLE",
        "OFF-WHITE",
        "YELLOW",
        "CHARCOAL",
        "TURQUOISE",
        "PINK",
        "LIME"
]



interior_options = [ 
        "BLACK",
        "GRAY",
        "BEIGE",
        "TAN",
        "—",
        "BROWN",
        "RED",
        "SILVER",
        "BLUE",
        "OFF-WHITE",
        "GOLD",
        "PURPLE",
        "WHITE",
        "GREEN",
        "BURGUNDY",
        "ORANGE",
        "YELLOW"
]


transmission_options = ['Manual', 'Automatic']



def show_predict_page():

    st.title("Car Price Estimator")

    st.write("""### We need some information to predict the price of your car""")

    # Define models for each make


    make = st.selectbox("Brand", list(models_by_make.keys()))
    model_options = models_by_make.get(make, [])
    model = st.selectbox("Model", model_options)  # Populate with models for the selected make

    # Convert make and model to integer labels
    make_label = le_make.fit_transform([make])[0]
    model_label = le_model.fit_transform([model])[0]

    #body = st.selectbox("Body", body)
    color = st.selectbox("Color", color_options)
    interior = st.selectbox("Interior", interior_options)
    transmission = st.selectbox("Transmission", transmission_options)
    odometer = st.slider("Mileage", 0, 999999, 0)
    condition = st.slider("Condition", 0, 50, 30)
    year = st.slider("Year", 1990, 2015, 2010)

    ok = st.button("Estimate Price")
    if ok:
        X = np.array([[year, make, model, transmission, condition, odometer, color, interior]])

        print("This is X inputs from the user", X)
        X[:, 1] = make_label
        X[:, 2] = model_label
        X[:, 3] = le_transmission.fit_transform(X[:,3])
        X[:, 6] = le_color.fit_transform(X[:,6])
        X[:, 7] = le_interior.fit_transform(X[:,7])
        X = X.astype(float)

        sellingprice = reg.predict(X)

        # Calculate lower and upper bounds of the range and round to the nearest integer
        lower_bound = round(sellingprice[0] * 0.9)
        upper_bound = round(sellingprice[0] * 1.1)

        # Convert the values to integers
        lower_bound = int(lower_bound)
        upper_bound = int(upper_bound)

        
        st.markdown(f"The estimated price for your car is somewhere between **${lower_bound}** and **${upper_bound}**", unsafe_allow_html=True)


show_predict_page()