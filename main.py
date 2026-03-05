import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import json
from datetime import datetime
from collections import defaultdict

# Set page config
st.set_page_config(page_title="Plant Disease Detector 🌿", page_icon="🩺", layout="wide")

# Load Lottie animation from URL or file
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

# Disease classes (matching model)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Language-specific labels
labels = {
    "English": {
        "predict": "Predict",
        "class": "Detected Class:",
        "confidence": "Confidence:",
        "recommendation": "Recommendation:",
        "upload_prompt": "Upload a leaf image",
        "subtitle_morning": "Good morning! Let's check your plant's health 🌞",
        "subtitle_afternoon": "Good afternoon! Time for a plant health check 🌤️",
        "subtitle_evening": "Good evening! Analyze your plant's condition 🌙"},
    "Telugu": {
        "predict": "అంచనా వేయి",
        "class": "కనుగొన్న తరగతి:",
        "confidence": "నిర్ధారిత నమ్మకం:",
        "recommendation": "సిఫార్సు:",
        "upload_prompt": "ఒక ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "subtitle_morning": "శుభోదయం! మొక్క ఆరోగ్యాన్ని పరీక్షిద్దాం 🌞",
        "subtitle_afternoon": "మధ్యాహ్నం శుభాకాంక్షలు! ఆరోగ్య తనిఖీ చేద్దాం 🌤️",
        "subtitle_evening": "శుభసాయంత్రం! మీ మొక్క స్థితిని విశ్లేషించండి 🌙"},
    "Hindi": {
        "predict": "पूर्वानुमान करें",
        "class": "पता चला वर्ग:",
        "confidence": "विश्वास:",
        "recommendation": "सिफारिश:",
        "upload_prompt": "एक पत्ती की छवि अपलोड करें",
        "subtitle_morning": "सुप्रभात! चलिए पौधे की सेहत जांचते हैं 🌞",
        "subtitle_afternoon": "नमस्कार! पौधे का स्वास्थ्य जांचें 🌤️",
        "subtitle_evening": "शुभ संध्या! अपने पौधे की स्थिति का विश्लेषण करें 🌙"}
}

# Determine subtitle by time
def get_time_based_subtitle(lang):
    hour = datetime.now().hour
    if hour < 12:
        return labels[lang]["subtitle_morning"]
    elif hour < 17:
        return labels[lang]["subtitle_afternoon"]
    else:
        return labels[lang]["subtitle_evening"]

# Disease Recommendations (from main3)
disease_recommendations = defaultdict(lambda: {
    'English': 'Consult a local expert for treatment.',
    'Telugu': 'చికిత్స కోసం స్థానిక నిపుణుడిని సంప్రదించండి.',
    'Hindi': 'उपचार के लिए स्थानीय विशेषज्ञ से परामर्श करें।'
}, {
    'Apple___Apple_scab': {
        'English': 'Apply fungicides like captan or myclobutanil. Prune infected branches.',
        'Telugu': 'క్యాప్టాన్ లేదా మైక్లోబుటానిల్ వంటి ఫంగిసైడ్లు వాడండి. బాధిత శాఖలను త్రిమ్మండి.',
        'Hindi': 'कैप्टन या माइकलोबुटानिल जैसे फफूंदनाशकों का प्रयोग करें। संक्रमित शाखाओं को छाँटें।'
    },
    'Apple___Black_rot': {
        'English': 'Remove infected fruits and apply fungicides like thiophanate-methyl.',
        'Telugu': 'బాధిత ఫలాలను తొలగించి థయోఫెనేట్-మెథిల్ వంటి ఫంగిసైడ్లు వాడండి.',
        'Hindi': 'संक्रमित फलों को हटाएं और थायोफेनेट-मेथिल जैसे फफूंदनाशकों का उपयोग करें।'
    },
    'Apple___Cedar_apple_rust': {
        'English': 'Remove nearby juniper trees. Use myclobutanil.',
        'Telugu': 'సమీపంలోని జునిపర్ చెట్లను తొలగించండి. మైక్లోబుటానిల్ వాడండి.',
        'Hindi': 'पास के जुनिपर पेड़ों को हटा दें। माइकलोबुटानिल का प्रयोग करें।'
    },
    'Apple___healthy': {
        'English': 'No disease detected. Keep monitoring!',
        'Telugu': 'వ్యాధి కనిపించలేదు. పర్యవేక్షణ కొనసాగించండి!',
        'Hindi': 'कोई बीमारी नहीं पाई गई। निगरानी बनाए रखें!'
    },
    'Blueberry___healthy': {
        'English': 'Blueberry plant is healthy. Keep up the good work!',
        'Telugu': 'బ్లూబెర్రీ మొక్క ఆరోగ్యంగా ఉంది. ఈ విధంగా కొనసాగించండి!',
        'Hindi': 'ब्लूबेरी का पौधा स्वस्थ है। ऐसे ही देखभाल जारी रखें!'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'English': 'Use fungicides like sulfur or potassium bicarbonate. Improve air circulation.',
        'Telugu': 'సల్ఫర్ లేదా పొటాషియం బైకార్బోనేట్ వంటి ఫంగిసైడ్లు వాడండి. గాలి ప్రసరణ మెరుగుపరచండి.',
        'Hindi': 'सल्फर या पोटैशियम बाइकार्बोनेट जैसे फफूंदनाशकों का उपयोग करें। हवा का प्रवाह सुधारें।'
    },
    'Cherry_(including_sour)___healthy': {
        'English': 'Cherry plant is healthy. Maintain regular care.',
        'Telugu': 'చెర్రీ మొక్క ఆరోగ్యంగా ఉంది. సాధారణ సంరక్షణ కొనసాగించండి.',
        'Hindi': 'चेरी का पौधा स्वस्थ है। नियमित देखभाल बनाए रखें।'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'English': 'Apply fungicides. Rotate crops. Use disease-resistant varieties.',
        'Telugu': 'ఫంగిసైడ్లు వాడండి. పంటలు మార్చండి. వ్యాధి నిరోధక వేరుశెనగ రకాలు ఉపయోగించండి.',
        'Hindi': 'फफूंदनाशकों का प्रयोग करें। फसलों का रोटेशन करें। रोग प्रतिरोधी किस्में लगाएं।'
    },
    'Corn_(maize)___Common_rust_': {
        'English': 'Use resistant hybrids and apply fungicides if needed.',
        'Telugu': 'నిరోధక హైబ్రిడ్ రకాలను వాడండి. అవసరమైతే ఫంగిసైడ్లు వాడండి.',
        'Hindi': 'प्रतिरोधी किस्में लगाएं और आवश्यकता होने पर फफूंदनाशकों का प्रयोग करें।'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'English': 'Use resistant hybrids. Apply fungicides as needed.',
        'Telugu': 'నిరోధక రకాలను వాడండి. అవసరమైతే ఫంగిసైడ్లు వాడండి.',
        'Hindi': 'प्रतिरोधी किस्में लगाएं। जरूरत पड़ने पर फफूंदनाशकों का उपयोग करें।'
    },
    'Corn_(maize)___healthy': {
        'English': 'Corn leaf is healthy. No action needed.',
        'Telugu': 'మొక్క ఆరోగ్యంగా ఉంది. చర్య అవసరం లేదు.',
        'Hindi': 'मक्का की पत्ती स्वस्थ है। कोई कार्रवाई आवश्यक नहीं है।'
    },
    'Grape___Black_rot': {
        'English': 'Prune infected parts. Apply fungicides like myclobutanil.',
        'Telugu': 'బాధిత భాగాలను త్రిమ్మండి. మైక్లోబుటానిల్ వంటి ఫంగిసైడ్లు వాడండి.',
        'Hindi': 'संक्रमित भागों को छाँटें। माइकलोबुटानिल जैसे फफूंदनाशकों का प्रयोग करें।'
    },
    'Grape___Esca_(Black_Measles)': {
        'English': 'Prune and destroy infected wood. Avoid injuries.',
        'Telugu': 'బాధిత కొమ్మలను త్రిమ్మి నశింపజేయండి. గాయాలు తలెత్తకుండా చూసుకోండి.',
        'Hindi': 'संक्रमित लकड़ी को काटकर नष्ट करें। चोटों से बचें।'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'English': 'Apply copper-based fungicides. Manage canopy for airflow.',
        'Telugu': 'కాపర్ ఆధారిత ఫంగిసైడ్లు వాడండి. గాలి ప్రసరణ కోసం కాపీ నిర్వహించండి.',
        'Hindi': 'कॉपर आधारित फफूंदनाशकों का उपयोग करें। हवा के प्रवाह के लिए छाया प्रबंधन करें।'
    },
    'Grape___healthy': {
        'English': 'Grape vine is healthy. Maintain current practices.',
        'Telugu': 'ద్రాక్ష మొక్క ఆరోగ్యంగా ఉంది. ప్రస్తుత నిర్వహణ కొనసాగించండి.',
        'Hindi': 'अंगूर का पौधा स्वस्थ है। वर्तमान देखभाल बनाए रखें।'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'English': 'Remove infected trees. Control psyllid insects using insecticides.',
        'Telugu': 'బాధిత చెట్లను తొలగించండి. పురుగు మందులతో సైల్లిడ్ పురుగులను నియంత్రించండి.',
        'Hindi': 'संक्रमित पेड़ों को हटाएं। कीटनाशकों से साइलिड कीटों को नियंत्रित करें।'
    },
    'Peach___Bacterial_spot': {
        'English': 'Apply copper sprays. Remove affected leaves and fruits.',
        'Telugu': 'కాపర్ స్ప్రేలు వాడండి. బాధిత ఆకులు మరియు పండ్లను తొలగించండి.',
        'Hindi': 'कॉपर स्प्रे का उपयोग करें। संक्रमित पत्तियों और फलों को हटाएं।'
    },
    'Peach___healthy': {
        'English': 'Peach plant is healthy. Continue standard care.',
        'Telugu': 'పీచ్ మొక్క ఆరోగ్యంగా ఉంది. సాధారణ సంరక్షణ కొనసాగించండి.',
        'Hindi': 'आड़ू का पौधा स्वस्थ है। सामान्य देखभाल जारी रखें।'
    },
    'Pepper,_bell___Bacterial_spot': {
        'English': 'Use copper fungicides. Avoid overhead watering.',
        'Telugu': 'కాపర్ ఫంగిసైడ్లు వాడండి. పై నుంచి నీరు పోయడం నివారించండి.',
        'Hindi': 'कॉपर फफूंदनाशकों का उपयोग करें। ऊपर से पानी देना टालें।'
    },
    'Pepper,_bell___healthy': {
        'English': 'Bell pepper plant is healthy.',
        'Telugu': 'బెల్ పెప్పర్ మొక్క ఆరోగ్యంగా ఉంది.',
        'Hindi': 'शिमला मिर्च का पौधा स्वस्थ है।'
    },
    'Potato___Early_blight': {
        'English': 'Apply fungicides like chlorothalonil. Remove plant debris.',
        'Telugu': 'క్లోరోథాలోనిల్ వంటి ఫంగిసైడ్లు వాడండి. మొక్క మలిని తొలగించండి.',
        'Hindi': 'क्लोरोथालोनिल जैसे फफूंदनाशकों का प्रयोग करें। पौधों का मलबा हटाएं।'
    },
    'Potato___Late_blight': {
        'English': 'Use fungicides like metalaxyl. Destroy infected plants.',
        'Telugu': 'మెటలాక్సిల్ వంటి ఫంగిసైడ్లు వాడండి. బాధిత మొక్కలను నాశనం చేయండి.',
        'Hindi': 'मेटालैक्सिल जैसे फफूंदनाशकों का प्रयोग करें। संक्रमित पौधों को नष्ट करें।'
    },
    'Potato___healthy': {
        'English': 'Potato plant is healthy. No action needed.',
        'Telugu': 'అలూగడ్డ మొక్క ఆరోగ్యంగా ఉంది. చర్య అవసరం లేదు.',
        'Hindi': 'आलू का पौधा स्वस्थ है। कोई कार्रवाई आवश्यक नहीं है।'
    },
    'Raspberry___healthy': {
        'English': 'Raspberry plant is healthy.',
        'Telugu': 'రాస్ప్బెర్రీ మొక్క ఆరోగ్యంగా ఉంది.',
        'Hindi': 'रास्पबेरी का पौधा स्वस्थ है।'
    },
    'Soybean___healthy': {
        'English': 'Soybean plant is healthy.',
        'Telugu': 'సోయాబీన్ మొక్క ఆరోగ్యంగా ఉంది.',
        'Hindi': 'सोयाबीन का पौधा स्वस्थ है।'
    },
    'Squash___Powdery_mildew': {
        'English': 'Apply fungicides like sulfur. Improve air circulation.',
        'Telugu': 'సల్ఫర్ వంటి ఫంగిసైడ్లు వాడండి. గాలి ప్రసరణ మెరుగుపరచండి.',
        'Hindi': 'सल्फर जैसे फफूंदनाशकों का प्रयोग करें। हवा का प्रवाह सुधारें।'
    },
    'Strawberry___Leaf_scorch': {
        'English': 'Remove infected leaves. Apply fungicides like captan.',
        'Telugu': 'బాధిత ఆకులను తొలగించండి. క్యాప్టాన్ వంటి ఫంగిసైడ్లు వాడండి.',
        'Hindi': 'संक्रमित पत्तियों को हटाएं। कैप्टन जैसे फफूंदनाशकों का प्रयोग करें।'
    },
    'Strawberry___healthy': {
        'English': 'Strawberry plant is healthy.',
        'Telugu': 'స్ట్రాబెర్రీ మొక్క ఆరోగ్యంగా ఉంది.',
        'Hindi': 'स्ट्रॉबेरी का पौधा स्वस्थ है।'
    },
    'Tomato___Bacterial_spot': {
        'English': 'Use copper sprays. Remove infected plants.',
        'Telugu': 'కాపర్ స్ప్రేలు వాడండి. బాధిత మొక్కలను తొలగించండి.',
        'Hindi': 'कॉपर स्प्रे का प्रयोग करें। संक्रमित पौधों को हटा दें।'
    },
    'Tomato___Early_blight': {
        'English': 'Apply fungicides like chlorothalonil. Rotate crops.',
        'Telugu': 'క్లోరోథాలోనిల్ వంటి ఫంగిసైడ్లు వాడండి. పంటల మార్పిడి చేయండి.',
        'Hindi': 'क्लोरोथालोनिल जैसे फफूंदनाशकों का उपयोग करें। फसल चक्र अपनाएं।'
    },
    'Tomato___Late_blight': {
        'English': 'Use metalaxyl. Destroy infected plants.',
        'Telugu': 'మెటలాక్సిల్ వాడండి. బాధిత మొక్కలను నాశనం చేయండి.',
        'Hindi': 'मेटालैक्सिल का प्रयोग करें। संक्रमित पौधों को नष्ट करें।'
    },
    'Tomato___Leaf_Mold': {
        'English': 'Avoid overhead watering. Increase airflow.',
        'Telugu': 'పైనుండి నీటిని పోయడాన్ని నివారించండి. గాలి ప్రసరణ పెంచండి.',
        'Hindi': 'ऊपर से पानी देना टालें। हवा का प्रवाह बढ़ाएं।'
    },
    'Tomato___Septoria_leaf_spot': {
        'English': 'Use fungicides. Remove infected leaves.',
        'Telugu': 'ఫంగిసైడ్లు వాడండి. బాధిత ఆకులను తొలగించండి.',
        'Hindi': 'फफूंदनाशकों का उपयोग करें। संक्रमित पत्तियों को हटाएं।'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'English': 'Spray water or use neem oil. Control spider mites.',
        'Telugu': 'నీటిని స్ప్రే చేయండి లేదా నిమ్ ఆయిల్ వాడండి. మైట్స్ నియంత్రించండి.',
        'Hindi': 'पानी छिड़कें या नीम का तेल प्रयोग करें। माइट्स को नियंत्रित करें।'
    },
    'Tomato___Target_Spot': {
        'English': 'Remove infected parts. Use fungicides.',
        'Telugu': 'బాధిత భాగాలను తొలగించండి. ఫంగిసైడ్లు వాడండి.',
        'Hindi': 'संक्रमित भागों को हटाएं। फफूंदनाशकों का प्रयोग करें।'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'English': 'Control whiteflies. Use resistant varieties.',
        'Telugu': 'తెల్ల దోమలను నియంత్రించండి. నిరోధక రకాలను వాడండి.',
        'Hindi': 'सफेद मक्खियों को नियंत्रित करें। प्रतिरोधी किस्में लगाएं।'
    },
    'Tomato___Tomato_mosaic_virus': {
        'English': 'Remove infected plants. Disinfect tools.',
        'Telugu': 'బాధిత మొక్కలను తొలగించండి. సాధనాలను శుద్ధి చేయండి.',
        'Hindi': 'संक्रमित पौधों को हटाएं। उपकरणों को कीटाणुरहित करें।'
    },
    'Tomato___healthy': {
        'English': 'Tomato plant is healthy.',
        'Telugu': 'టమోటా మొక్క ఆరోగ్యంగా ఉంది.',
        'Hindi': 'टमाटर का पौधा स्वस्थ है।'
    }
})
# Load animation
lottie_plant = load_lottie_file("plant.json")

# Language selection
language = st.sidebar.selectbox("Choose Language / భాషను ఎంచుకోండి / भाषा चुनें", ["English", "Telugu", "Hindi"])

# Title and animation
with st.container():
    st.title("🩺 Plant Disease Detector 🌿")
    st.subheader(get_time_based_subtitle(language))
    st_lottie(lottie_plant, height=250)

# File uploader
uploaded_file = st.file_uploader(labels[language]["upload_prompt"], type=["jpg", "jpeg", "png"])

# Helper for severity color/icon
def get_severity_display(predicted_class):
    if "healthy" in predicted_class.lower():
        return "🟢 Healthy"
    elif "rust" in predicted_class.lower() or "blight" in predicted_class.lower():
        return "🔴 Severe"
    elif "spot" in predicted_class.lower() or "mildew" in predicted_class.lower():
        return "🟠 Moderate"
    else:
        return "🟡 Mild"

# Prediction function (mirroring main3.py logic)
def model_prediction(image_file):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # No normalization if main3 didn’t use it
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_class_index, confidence

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    if st.button(labels[language]["predict"]):
        result_index, predicted_confidence = model_prediction(uploaded_file)
        predicted_class = class_names[result_index]
        severity_label = get_severity_display(predicted_class)

        st.markdown(f"### {labels[language]['class']} {severity_label}")
        st.success(predicted_class)

        st.markdown(f"### {labels[language]['confidence']} 🎯")
        st.info(f"{predicted_confidence * 100:.2f}%")

        st.markdown(f"### {labels[language]['recommendation']} 📋")
        with st.expander("View Recommendation"):
            st.warning(disease_recommendations[predicted_class].get(language, disease_recommendations[predicted_class]['English']))
