
import streamlit as st
from gtts import gTTS
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
import base64
import io
import numpy as np
import tensorflow as tf
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment








# === Set Fullscreen Background ===
def set_fullscreen_background():
    try:
        image_path = "PF.jpg"  # Rename your image file to this
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()

        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .stMarkdown, .stButton>button {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        st.error("⚠️ Could not set full background.")
# === Set Background Image (Allowing User Upload) ===
def set_bg_from_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            encoded_string = base64.b64encode(bytes_data).decode()
            mime_type = uploaded_file.type

            css = f"""
            <style>
            .stApp {{
                background-image: url("data:{mime_type};base64,{encoded_string}");
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center center;
                background-attachment: fixed;
            }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error setting background image from uploaded file: {e}")
    else:
        default_image_path = "PF.jpg"
        if os.path.exists(default_image_path):
            try:
                with open(default_image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                css = f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpg;base64,{encoded_string}");
                    background-repeat: no-repeat;
                    background-size: cover;
                    background-position: center center;
                    background-attachment: fixed;
                }}
                </style>
                """
                st.markdown(css, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not load default background image '{default_image_path}': {e}")
                st.markdown(
                    """
                    <style>
                    .stApp {
                        background-color: #f0f2f6;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #f0f2f6;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

        

# === Set Background Image ===
def set_top_banner():
    try:
        banner_path = "background.jpg"  # Your predefined banner image (should be ~100–150px tall)

        with open(banner_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()

        css = f"""
        <style>
        .top-banner {{
            width: 100%;
            height: 140px;
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 1px 1px 3px #000;
            margin-bottom: 20px;
            border-radius: 6px;
        }}
        </style>
        <div class="top-banner">
            Smart Rice Classifier With Real-Time Prediction
        </div>
        """
        st.markdown(css, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Could not load banner image.")



# === Page Setup ===
st.set_page_config(page_title="🌾 Smart Rice Classifier", layout="wide")
# === Page Setup ===


#st.set_page_config(layout="wide")
set_fullscreen_background()  # ✅ Call it right after page config
#def set_fullscreen_background():
    


# === Load CSV ===
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except:
        return pd.DataFrame()

csv_file_path = "rice_types_unique_detailed.csv"
df_data = load_data(csv_file_path)
if df_data.empty:
    st.error("❌ Failed to load rice dataset.")
    st.stop()

# === Load CNN Model ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_cnn_model.h5")

cnn_model = load_model()

# === Class Label Mapping (order matters) ===
class_names = ['10_LalAush', '11_Jirashail', '12_Gutisharna', '13_RedCargo', '14_Najirshail', '15_KatariPolao', '16_LalBiroi', '17_ChiniguraPolao', '18Amon', '19_Shorna5', '1_SubolLota', '20_LalBinni', '2_Bashmoti', '3_Ganjiya', '4_Shampakatari', '5_Katarivog', '6_BR28', '7_BR29', '8_Paijam', '9_Bashful']

# === Sidebar ===

#st.sidebar.title("🔍 Rice Classifier Controls")
#set_top_banner()
# === Sidebar ===

# === Sidebar Logo Display Function ===
def show_sidebar_logo():
    try:
        logo_path = "rice_pre.jpg"  # Make sure this file is in the same folder
        with open(logo_path, "rb") as f:
            encoded_logo = base64.b64encode(f.read()).decode()

        logo_html = f"""
        <div style="text-align: center; margin-top: 5px; margin-bottom: 5px;">
            <img src="data:image/jpeg;base64,{encoded_logo}" style="width: 100px; height: auto; border-radius: 8px;" />
            <div style="font-size: 16px; font-weight: bold; color: #4CAF50;">Paddy Predictor</div>
        </div>
        """
        st.sidebar.markdown(logo_html, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error("⚠️ Could not load logo.")



        

# === Call Sidebar Logo Function ===
show_sidebar_logo()
st.sidebar.title("🔍 Rice Classifier Controls")
set_top_banner()


# === Upload & Set Background Image ===
bg_image = st.sidebar.file_uploader("♻️ Upload Background Image", type=["jpg", "jpeg", "png"])

# === Style the file uploader button to black ===
st.markdown("""
    <style>
    section[data-testid="stFileUploader"] button {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 8px;
        padding: 6px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# === Set Background if Uploaded ===
if bg_image is not None:
    try:
        encoded = base64.b64encode(bg_image.read()).decode()
        mime_type = bg_image.type
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:{mime_type};base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center center;
                background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ Failed to set background: {e}")


#audio controls
st.sidebar.subheader("🗣 Audio Language")
audio_language = st.sidebar.selectbox("🎙️ Select Audio Language", ["English", "Hindi", "Bengali"])
lang_map = {"English": "en", "Hindi": "hi", "Bengali": "bn"}
selected_lang_code = lang_map.get(audio_language, "en")


uploaded_rice_image = st.sidebar.file_uploader("📸 Upload Rice Image", type=["jpg", "png", "jpeg"])

# === Sidebar Footer ===
st.sidebar.markdown("""
<hr style="margin-top: 20px;">

<div style='text-align: center; font-size: 13px; color: gray;'>
      🧑‍💼Developed by <b>Paddy Predictor Team © 2025</b><br>
       <b>  </b><br>
    
</div>
""", unsafe_allow_html=True)


def render_sidebar(predicted_variety, confidence, info=None):
    with st.sidebar:
        st.markdown("## 🌾 Prediction Summary")
        st.markdown(f"<div style='font-size:20px; font-weight:bold;'>🔍 Predicted: <span style='color:#4FC3F7'>{predicted_variety}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:16px;'>📊 Confidence: <b>{confidence*100:.2f}%</b></div>", unsafe_allow_html=True)
        
        if info is not None:
            st.markdown("---", unsafe_allow_html=True)
            st.markdown("## 🌱 <span style='color:#2ECC71'>Cultivation & Environment</span>", unsafe_allow_html=True)

            info_box_style = "padding: 8px 16px; margin: 8px 0; background-color: #f9f9f9; border-radius: 8px; box-shadow: 1px 1px 5px rgba(0,0,0,0.05); font-size: 15px;"

            st.markdown(f"<div style='{info_box_style}'>🧬 <b>Scientific Name:</b> {info.get('Scientific Name', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='{info_box_style}'>🌿 <b>Cultivation Type:</b> {info.get('Cultivation & Growth', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='{info_box_style}'>🌍 <b>Soil Type:</b> {info.get('Soil', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='{info_box_style}'>🛡️ <b>Pest Resistance:</b> {info.get('Pest Resistant', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='{info_box_style}'>🧫 <b>Disease Resistance:</b> {info.get('Disease Resistance Level', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='{info_box_style}'>💰 <b>Market Price:</b> ₹{info.get('Market Price (Rs/kg)', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='{info_box_style}'>🔥 <b>Demand:</b> {info.get('Demand Level', 'N/A')}</div>", unsafe_allow_html=True)

        else:
            st.warning("No environmental details found.")

        # Audio Summary
        audio_text = f"{predicted_variety} rice has {info.get('Protein (g)', 'N/A')} grams of protein and is mainly harvested in {info.get('Harvesting', 'N/A')}."
        lang_map = {"English": "en", "Hindi": "hi", "Bengali": "bn"}
        try:
            tts = gTTS(audio_text, lang=lang_map.get(audio_language, "en"))
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format='audio/mp3')
        except Exception as e:
            st.error("🔊 Audio generation failed")

# === Real-Time Prediction ===
if uploaded_rice_image is not None:
    st.subheader("🤖 AI-Powered Real-Time Prediction")
    image = Image.open(uploaded_rice_image).convert("RGB")
    st.image(image, caption="Uploaded Rice Image", use_container_width=True)

    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    

    # Prediction result
    prediction = cnn_model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_variety = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    st.success(f"✅ Predicted Rice Variety: **{predicted_variety}**")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")

    # === Fix label to match CSV ===
    predicted_clean_name = "_".join(predicted_variety.split("_")[1:]).replace("_", " ").strip().lower()

    # Fetch info from CSV
    predicted_info = df_data[df_data["Rice Name"].str.lower().str.strip() == predicted_clean_name]

    if predicted_info is not None:
        info = predicted_info.iloc[0]
        st.markdown("## 📋<span style='color:#27ae60'> Rice Variety Details", unsafe_allow_html=True)

        card_style = """
        <style>
        .detail-card {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 16px;
        }
        .card-item {
            flex: 1 1 48%;
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 15px;
        }
        .card-item b {
            color: #2c3e50;
        }
        </style>
        """
        st.markdown(card_style, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="detail-card">
            <div class="card-item"><b>Scientific Name:</b> {info.get('Scientific Name', 'N/A')}</div>
            <div class="card-item"><b>Also Known As:</b> {info.get('Also Known As', 'N/A')}</div>
            <div class="card-item"><b>Storage Advice:</b> {info.get('Storage Advice', 'N/A')}</div>
            <div class="card-item"><b>Competing Varieties:</b> {info.get('Competing Varieties', 'N/A')}</div>
            <div class="card-item"><b>Cultivation & Growth:</b> {info.get('Cultivation & Growth', 'N/A')}</div>
            <div class="card-item"><b>Soil:</b> {info.get('Soil', 'N/A')}</div>
            <div class="card-item"><b>Nutritional Benefits:</b> {info.get('Nutritional Benefits', 'N/A')}</div>
            <div class="card-item"><b>Plant Height:</b> {info.get('Plant Height', 'N/A')}</div>
            <div class="card-item"><b>Structure of Rice:</b> {info.get('Structure of Rice', 'N/A')}</div>
            <div class="card-item"><b>Falling with Grain:</b> {info.get('Falling with Grain', 'N/A')}</div>
            <div class="card-item"><b>Color:</b> {info.get('Color', 'N/A')}</div>
            <div class="card-item"><b>Aroma:</b> {info.get('Aroma', 'N/A')}</div>
            <div class="card-item"><b>Genetic Diversity:</b> {info.get('Genetic Diversity', 'N/A')}</div>
            <div class="card-item"><b>Global Production:</b> {info.get('Global Production', 'N/A')}</div>
            <div class="card-item"><b>Environmental Impact:</b> {info.get('Environmental Impact', 'N/A')}</div>
            <div class="card-item"><b>Pest Resistant:</b> {info.get('Pest Resistant', 'N/A')}</div>
            <div class="card-item"><b>Disease Resistance:</b> {info.get('Disease Resistance Level', 'N/A')}</div>
            <div class="card-item"><b>Market Price (Rs/kg):</b> ₹{info.get('Market Price (Rs/kg)', 'N/A')}</div>
            <div class="card-item"><b>Demand Level:</b> {info.get('Demand Level', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("No environmental details found.")

    if not predicted_info.empty:
        st.markdown("## 🗓️ <span style='color:#27ae60'>Planting & Harvesting Schedule</span>", unsafe_allow_html=True)

        schedule_card_css = """
        <style>
        .schedule-card {
            background-color: #f4f9f4;
            border-radius: 10px;
            padding: 16px;
            margin-top: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
        .schedule-card b {
            color: #2c3e50;
            font-size: 15px;
        }
        .schedule-item {
            margin-bottom: 8px;
        }
        </style>
        """
        st.markdown(schedule_card_css, unsafe_allow_html=True)

        planting_season_str = info.get('Planting Season', "N/A")
        harvesting_time_str = info.get('Harvesting', "N/A")
        time_taken_str = info.get('Time Taken', "N/A")

        st.markdown(f"""
        <div class="schedule-card">
            <div class="schedule-item"><b>🌱 Planting Season:</b> {planting_season_str}</div>
            <div class="schedule-item"><b>🌾 Harvesting Time:</b> {harvesting_time_str}</div>
            <div class="schedule-item"><b>⏱️ Time Taken:</b> {time_taken_str}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🌿<span style='color:#27ae60'>Visualization of Planting & Harvesting Schedule - Conceptual</span>", unsafe_allow_html=True)
        # Simplified visualization of planting and harvesting months
        st.markdown("Note: This chart shows the approximate months for planting and harvesting based on the provided data. Note that actual months may vary based on local climate and conditions.")
        
        try:
            plant_month = 1
            harvest_month = 1

            def extract_month(text):
                month_map = {
                    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
                }
                for name, num in month_map.items():
                    if name in text.lower():
                        return num
                return 1

            plant_month = extract_month(planting_season_str)
            harvest_month = extract_month(harvesting_time_str)

            plot_data = pd.DataFrame({
                'Month': ['Planting', 'Harvesting'],
                'Value': [plant_month, harvest_month]
            })

            fig, ax = plt.subplots(figsize=(4, 2))
            ax.bar(plot_data['Month'], plot_data['Value'], color=['green', 'orange'])
            ax.set_ylabel('Approx Month (1-12)')
            ax.set_title('Simplified Planting and Harvesting Months')
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not generate planting/harvesting chart: {e}") 
        
        
            

        st.subheader("🌐 Regional Information")
        regional_info_raw = info.get("Region", "")
        regional_states = [s.strip() for s in regional_info_raw.split(',')] if isinstance(regional_info_raw, str) else []
        if regional_states:
            st.markdown(f"- <b>{predicted_variety} is associated with the following regions:</b> {', '.join(regional_states)}", unsafe_allow_html=True)
        else:
            st.markdown(f"- Regional information for {predicted_variety} not available in the dataset.", unsafe_allow_html=True)

    else:
        st.warning("ℹ️ No detailed info found in dataset.")



    # === Text-to-Speech (TTS) ===
    # pip install googletrans==4.0.0rc1
    from googletrans import Translator

    translator = Translator()

    if not predicted_info.empty:
        info = predicted_info.iloc[0]

        # English text
        audio_text = (
            f"{predicted_clean_name.title()} rice is known for {info.get('Nutritional Benefits', 'N/A')}. "
            f"It is grown mainly in {info.get('Cultivation & Growth', 'N/A')}. "
            f"Market price is around ₹{info.get('Market Price (Rs/kg)', 'N/A')} per kg."
        )

        # Translate if language is not English
        selected_lang_code = {"English": "en", "Hindi": "hi", "Bengali": "bn"}.get(audio_language, "en")
        if selected_lang_code != "en":
            try:
                translated = translator.translate(audio_text, dest=selected_lang_code)
                audio_text = translated.text
            except Exception as e:
                st.warning("⚠️ Translation failed. Defaulting to English.")

        # Generate audio
        try:
            tts = gTTS(text=audio_text, lang=selected_lang_code)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format='audio/mp3')
        except Exception as e:
            st.warning(f"⚠️ Could not generate audio in {audio_language}.")  



# Your main app content above...

# Footer at the bottom of the page
st.markdown("""
<hr style="margin-top: 50px;">
<div style='text-align: center; font-size: px; color: gray;'>
    ✒️ Developed by <b>Paddy Predictor Team © 2025  From NSTI (W) Kolkata</b>
    👩‍💻 Team Members:<br>
     Pritha Roy<br>
     Rose Mary Rai<br>
    Moumita Shaha<br>
    Amina Khatun
</div>
""", unsafe_allow_html=True)



            






