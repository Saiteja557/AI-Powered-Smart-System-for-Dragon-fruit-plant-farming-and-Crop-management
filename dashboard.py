# STREAMLIT CONFIG 
import streamlit as st
st.set_page_config(page_title="FarmAssist – Dragon Fruit AI", layout="wide")
import random
# SYSTEM IMPORTS 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
from PIL import Image
from gtts import gTTS
import tempfile
# LOAD MODELS 
disease_model  = tf.keras.models.load_model("models/disease_model_fast.h5")
ripeness_model = tf.keras.models.load_model("models/ripeness_model.keras")
grading_model  = tf.keras.models.load_model("models/grading_model.keras")
binary_model = tf.keras.models.load_model("models/dragonfruit_binary_model.h5")
#  GOOGLE VOICE FUNCTION 
def speak_text(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        audio_file = open(temp_file.name, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error("Voice error: " + str(e))
VOICE_LANG = {
    "English": "en",
    "తెలుగు": "te",
    "हिंदी": "hi"
}
# LANGUAGE CONTENT (UPDATED WITH MULTI-PAGE SUPPORT)
LANG = {
    "English": {
        "title": "🌱 FarmAssist – Dragon Fruit AI",
        "subtitle": "🚜 KrushiMitra AI – Helping Farmers Increase Yield & Income",
        "quote": "To help farmers achieve higher crop yield and better income 🌾",
        "menu": ["🏠 Home", "📤 Image Analysis", "🌾 Crop Care", "👨‍🌾 Farmer Guide", "ℹ️ About"],
        "home_cards": [
            ("🌿 Disease Detection", "Detect plant diseases early"),
            ("🍉 Ripeness Detection", "Identify correct harvest time"),
            ("⭐ Quality  Grading", "Get better market value")
        ],
        "upload": "Upload Dragon Fruit / Leaf Image",
        "drag": "Drag & drop image here or browse",
        "results": "🔍 Analysis Results",
        "disease": " Disease Status",
        "ripeness": " Ripeness Status",
        "grade": " Quality Grade",
        "healthy": "Healthy",
        "diseased": "Diseased",
        "ripe": "Ripe",
        "unripe": "Unripe",
        "gradeA": "Grade A",
        "gradeB": "Grade B",
        "gradeC": "Grade C",
        "rec_title": "💊 Recommendation System",
        "rec_medicine": "🧪 Medicine",
        "rec_dosage": "⚖ Dosage",
        "rec_organic": "🌿 Organic Alternative",
        "rec_prevention": "🛡 Preventive Measures",
        "care_title": "🌾 Crop Protection Tips",
        "care_points": [
            "Regular field monitoring",
            "Remove infected plants",
            "Proper irrigation",
            "Use organic pesticides"
        ],
        "guide_title": "👨‍🌾 Farmer Guide",
        "guide_text": "Upload crop images regularly to protect yield.",
        "about": "FarmAssist is an AI-based smart agriculture system.",

        "best_practices": "Best Practices",
        "practice1": "Maintain soil health",
        "practice2": "Monitor pests weekly",
        "practice3": "Use balanced fertilizers",

        "feature1": "Disease Detection",
        "feature2": "Ripeness Prediction",
        "feature3": "Quality Grading",
        "feature4": "Smart Recommendation System",
        "feature5": "Multi-language Voice Support",

        "step1": "Upload crop image clearly under sunlight.",
        "step2": "Check AI detection results carefully.",
        "step3": "Follow recommended treatment immediately.",
        "step4": "Repeat monitoring every 7 days.",
        "guide_tip": "Early detection = Higher Profit",

        "about_title": "About FarmAssist",
        "features": "Features",
        "mission": "Mission",
        "mission_text": "To help farmers increase yield, reduce losses, and improve income using AI technology.",
        "developed_text": "Developed as an AI-based Smart Agriculture Project."
    },

    "తెలుగు": {
        "title": "🌱 ఫామ్ అసిస్ట్ – డ్రాగన్ ఫ్రూట్ AI",
        "subtitle": "🚜 కృషి మిత్ర AI – దిగుబడి & ఆదాయం పెంచేందుకు",
        "quote": "రైతుల ఆదాయం పెంచడమే మా లక్ష్యం 🌾",
        "menu": ["🏠 హోమ్", "📤 చిత్రం విశ్లేషణ", "🌾 పంట సంరక్షణ", "👨‍🌾 రైతు మార్గదర్శిని", "ℹ️ సమాచారం"],
        "home_cards": [
            ("🌿 రోగ గుర్తింపు", "ముందుగానే రోగాన్ని గుర్తించండి"),
            ("🍉 పక్వ గుర్తింపు", "సరైన కోత సమయం"),
            ("⭐ నాణ్యత గ్రేడింగ్", "మంచి ధర పొందండి")
        ],
        "upload": "డ్రాగన్ ఫ్రూట్ / ఆకు చిత్రం అప్‌లోడ్ చేయండి",
        "drag": "చిత్రాన్ని ఇక్కడ ఉంచండి",
        "results": "🔍 విశ్లేషణ ఫలితాలు",
        "disease": " రోగ స్థితి",
        "ripeness": " పక్వ స్థితి",
        "grade": " నాణ్యత గ్రేడ్",
        "healthy": "ఆరోగ్యంగా ఉంది",
        "diseased": "రోగం ఉంది",
        "ripe": "పక్వమైనది",
        "unripe": "పక్వం కాలేదు",
        "gradeA": "గ్రేడ్ A",
        "gradeB": "గ్రేడ్ B",
        "gradeC": "గ్రేడ్ C",
        "rec_title": "💊 సూచించిన చికిత్స",
        "rec_medicine": "🧪 మందు",
        "rec_dosage": "⚖ మోతాదు",
        "rec_organic": "🌿 సేంద్రీయ ప్రత్యామ్నాయం",
        "rec_prevention": "🛡 నివారణ చర్యలు",
        "care_title": "🌾 పంట రక్షణ సూచనలు",
        "care_points": [
            "పంటను తరచూ పరిశీలించండి",
            "రోగగ్రస్త మొక్కలను తొలగించండి",
            "సరైన నీటి నిర్వహణ"
        ],
        "guide_title": "👨‍🌾 రైతు మార్గదర్శిని",
        "guide_text": "పంట రక్షణకు చిత్రాలను తరచుగా అప్‌లోడ్ చేయండి.",
        "about": "ఫామ్ అసిస్ట్ AI ఆధారిత వ్యవసాయ వ్యవస్థ.",

        "best_practices": "ఉత్తమ పద్ధతులు",
        "practice1": "మట్టి ఆరోగ్యాన్ని కాపాడండి",
        "practice2": "ప్రతి వారం పురుగులను పరిశీలించండి",
        "practice3": "సమతుల్య ఎరువులు ఉపయోగించండి",

        "feature1": "రోగ గుర్తింపు",
        "feature2": "పక్వ అంచనా",
        "feature3": "నాణ్యత గ్రేడింగ్",
        "feature4": "స్మార్ట్ సూచన వ్యవస్థ",
        "feature5": "బహుభాషా వాయిస్ మద్దతు",

        "step1": "సూర్యకాంతిలో పంట చిత్రాన్ని స్పష్టంగా అప్‌లోడ్ చేయండి.",
        "step2": "AI ఫలితాలను జాగ్రత్తగా పరిశీలించండి.",
        "step3": "సిఫారసు చేసిన చికిత్సను వెంటనే పాటించండి.",
        "step4": "ప్రతి 7 రోజులకు పర్యవేక్షణ చేయండి.",
        "guide_tip": "ముందస్తు గుర్తింపు = ఎక్కువ లాభం",

        "about_title": "FarmAssist గురించి",
        "features": "ఫీచర్లు",
        "mission": "మిషన్",
        "mission_text": "AI సాంకేతికత ద్వారా రైతుల దిగుబడి మరియు ఆదాయాన్ని పెంచడం.",
        "developed_text": "AI ఆధారిత స్మార్ట్ వ్యవసాయ ప్రాజెక్ట్‌గా అభివృద్ధి చేయబడింది."
    },

    "हिंदी": {
        "title": "🌱 फार्मअसिस्ट – ड्रैगन फ्रूट AI",
        "subtitle": "🚜 कृषिमित्र AI – किसानों की मदद के लिए",
        "quote": "बेहतर फसल और अधिक आय 🌾",
        "menu": ["🏠 होम", "📤 छवि विश्लेषण", "🌾 फसल देखभाल", "👨‍🌾 किसान मार्गदर्शक", "ℹ️ जानकारी"],
        "home_cards": [
            ("🌿 रोग पहचान", "समय पर रोग पहचानें"),
            ("🍉 पकने की पहचान", "सही कटाई समय"),
            ("⭐ गुणवत्ता ग्रेडिंग", "बेहतर दाम")
        ],
        "upload": "ड्रैगन फ्रूट / पत्ती की छवि अपलोड करें",
        "drag": "यहाँ छवि डालें",
        "results": "🔍 विश्लेषण परिणाम",
        "disease": " रोग स्थिति",
        "ripeness": " पकने की स्थिति",
        "grade": " गुणवत्ता ग्रेड",
        "healthy": "स्वस्थ",
        "diseased": "रोगग्रस्त",
        "ripe": "पका हुआ",
        "unripe": "कच्चा",
        "gradeA": "ग्रेड A",
        "gradeB": "ग्रेड B",
        "gradeC": "ग्रेड C",
        "rec_title": "💊 अनुशंसित उपचार",
        "rec_medicine": "🧪 दवा",
        "rec_dosage": "⚖ मात्रा",
        "rec_organic": "🌿 जैविक विकल्प",
        "rec_prevention": "🛡 निवारक उपाय",
        "care_title": "🌾 फसल सुरक्षा सुझाव",
        "care_points": [
            "खेत निरीक्षण करें",
            "संक्रमित पौधे हटाएँ",
            "सही सिंचाई करें"
        ],
        "guide_title": "👨‍🌾 किसान मार्गदर्शक",
        "guide_text": "फसल सुरक्षा के लिए नियमित जाँच करें।",
        "about": "फार्मअसिस्ट स्मार्ट कृषि समाधान है।",

        "best_practices": "सर्वोत्तम अभ्यास",
        "practice1": "मिट्टी की सेहत बनाए रखें",
        "practice2": "हर सप्ताह कीट जांच करें",
        "practice3": "संतुलित उर्वरक उपयोग करें",

        "feature1": "रोग पहचान",
        "feature2": "पकने की भविष्यवाणी",
        "feature3": "गुणवत्ता ग्रेडिंग",
        "feature4": "स्मार्ट अनुशंसा प्रणाली",
        "feature5": "बहुभाषी वॉयस समर्थन",

        "step1": "धूप में फसल की साफ तस्वीर अपलोड करें।",
        "step2": "AI परिणाम ध्यान से देखें।",
        "step3": "सुझाए गए उपचार तुरंत अपनाएँ।",
        "step4": "हर 7 दिन में निगरानी करें।",
        "guide_tip": "जल्दी पहचान = अधिक लाभ",

        "about_title": "FarmAssist के बारे में",
        "features": "विशेषताएँ",
        "mission": "मिशन",
        "mission_text": "AI तकनीक से किसानों की आय और उत्पादन बढ़ाना।",
        "developed_text": "AI आधारित स्मार्ट कृषि परियोजना के रूप में विकसित।"
    }
}
# RECOMMENDATIONS
RECOMMENDATIONS = {
    "English": {
        "Diseased": [
            {
                "medicine": "Carbendazim",
                "dosage": "1 gram per liter of water",
                "organic": "Neem oil spray (5 ml per liter)",
                "prevention": [
                    "Remove infected plant parts",
                    "Avoid over-watering",
                    "Ensure good sunlight exposure"
                ]
            },
            {
                "medicine": "Mancozeb",
                "dosage": "2 grams per liter of water",
                "organic": "Garlic extract spray",
                "prevention": [
                    "Improve air circulation",
                    "Do not overcrowd plants",
                    "Use clean garden tools"
                ]
            },
            {
                "medicine": "Copper Oxychloride",
                "dosage": "2.5 grams per liter of water",
                "organic": "Trichoderma solution",
                "prevention": [
                    "Avoid water stagnation",
                    "Monitor plants weekly",
                    "Maintain proper drainage"
                ]
            }
        ]
    },

    "తెలుగు": {
        "Diseased": [
            {
                "medicine": "కార్బెండాజిమ్",
                "dosage": "లీటర్ నీటికి 1 గ్రాము",
                "organic": "వేప నూనె స్ప్రే (5 మిల్లీ / లీటర్)",
                "prevention": [
                    "రోగగ్రస్త భాగాలను తొలగించండి",
                    "అధిక నీరు పోయవద్దు",
                    "సూర్యకాంతి అందించండి"
                ]
            },
            {
                "medicine": "మ్యాంకోజెబ్",
                "dosage": "లీటర్ నీటికి 2 గ్రాములు",
                "organic": "వెల్లుల్లి సారం స్ప్రే",
                "prevention": [
                    "గాలి ప్రసరణ మెరుగుపరచండి",
                    "మొక్కలను దగ్గరగా నాటవద్దు",
                    "శుభ్రమైన పరికరాలు ఉపయోగించండి"
                ]
            },
            {
                "medicine": "కాపర్ ఆక్సీక్లోరైడ్",
                "dosage": "లీటర్ నీటికి 2.5 గ్రాములు",
                "organic": "ట్రైకోడెర్మా ద్రావణం",
                "prevention": [
                    "నీరు నిల్వ కాకుండా చూడండి",
                    "ప్రతి వారం పరిశీలించండి",
                    "సరైన డ్రైనేజ్ ఏర్పాటు చేయండి"
                ]
            }
        ]
    },

    "हिंदी": {
        "Diseased": [
            {
                "medicine": "कार्बेन्डाजिम",
                "dosage": "1 ग्राम प्रति लीटर पानी",
                "organic": "नीम तेल स्प्रे (5 ml प्रति लीटर)",
                "prevention": [
                    "संक्रमित भाग हटाएं",
                    "अधिक पानी न दें",
                    "पर्याप्त धूप दें"
                ]
            },
            {
                "medicine": "मैन्कोजेब",
                "dosage": "2 ग्राम प्रति लीटर पानी",
                "organic": "लहसुन अर्क स्प्रे",
                "prevention": [
                    "हवा का संचार बढ़ाएं",
                    "पौधों को पास-पास न लगाएं",
                    "साफ उपकरण इस्तेमाल करें"
                ]
            },
            {
                "medicine": "कॉपर ऑक्सी क्लोराइड",
                "dosage": "2.5 ग्राम प्रति लीटर पानी",
                "organic": "ट्राइकोडर्मा घोल",
                "prevention": [
                    "पानी जमा न होने दें",
                    "साप्ताहिक निरीक्षण करें",
                    "अच्छी जल निकासी रखें"
                ]
            }
        ]
    }
}
# IMAGE PREPROCESS 
def prepare_image(img, size):
    img = img.resize(size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

language = st.sidebar.selectbox("🌐Select Language / భాష / भाषा", list(LANG.keys()))
T = LANG[language]
menu = st.sidebar.radio("📋 Menu", T["menu"])

# HOME PAGE

if menu == T["menu"][0]:
    st.title(T["title"])
    st.subheader(T["subtitle"])
    st.info(T["quote"])
    c1, c2, c3 = st.columns(3)
    for col, card in zip([c1, c2, c3], T["home_cards"]):
        col.markdown(f"### {card[0]}")
        col.write(card[1])

# IMAGE ANALYSIS PAGE 
elif menu == T["menu"][1]:
    st.markdown("## 📤 " + T["upload"])
    st.markdown("---")
    file = st.file_uploader(T["drag"], type=["jpg", "jpeg", "png"])

    if file:
        col_img, col_space = st.columns([2,1])

        with col_img:
            img = Image.open(file).convert("RGB")
            st.image(img, use_container_width=True)

        # 🔹 STEP 1: Run Binary Classifier FIRST
        binary_pred = binary_model.predict(prepare_image(img, (128,128)))[0][0]

        if binary_pred > 0.5:
            st.error("⚠ Uploaded image is not related to Dragon Fruit. Please upload a valid dragon fruit image.")
            st.stop()

        # 🔹 STEP 2: If valid → run other models
        with st.spinner(" Analyzing Image with AI Model..."):
            d = disease_model.predict(prepare_image(img, (128, 128)))[0]
            r = ripeness_model.predict(prepare_image(img, (128, 128)))[0][0]
            g = grading_model.predict(prepare_image(img, (128, 128)))[0]

            disease = T["healthy"] if np.argmax(d) in [0, 2] else T["diseased"]
            ripeness = T["ripe"] if r >= 0.5 else T["unripe"]
            grade = [T["gradeA"], T["gradeB"], T["gradeC"]][np.argmax(g)]

        st.markdown("###  " + T["results"])
        col1, col2, col3 = st.columns(3)
        col1.metric(T["disease"], disease)
        col2.metric(T["ripeness"], ripeness)
        col3.metric(T["grade"], grade)

        st.progress(float(np.max(d)))

        summary = f"{T['disease']} {disease}. {T['ripeness']} {ripeness}. {T['grade']} {grade}."

        if st.button("🔊 Speak Results"):
            speak_text(summary, VOICE_LANG[language])
        
        # RECOMMENDATION SYSTEM 
        
        if disease == T["diseased"]:
            rec_list = RECOMMENDATIONS[language]["Diseased"]
            rec = random.choice(rec_list)
            st.markdown("---")
            st.markdown("##  " + T["rec_title"])
            with st.container():
                st.success(f"{T['rec_medicine']}: {rec['medicine']}")
                st.info(f"{T['rec_dosage']}: {rec['dosage']}")
                st.warning(f"{T['rec_organic']}: {rec['organic']}")
                st.markdown("### " + T["rec_prevention"])
                for tip in rec["prevention"]:
                    st.write("✔", tip)
            rec_text = f"{rec['medicine']}. {rec['dosage']}. {rec['organic']}."
            if st.button("🔊 Speak Treatment"):
                speak_text(rec_text, VOICE_LANG[language])    

# CROP CARE PAGE

elif menu == T["menu"][2]:
    st.markdown("## 🌾 " + T["care_title"])
    st.markdown("---")
    cols = st.columns(2)
    for i, point in enumerate(T["care_points"]):
        with cols[i % 2]:
            st.info("🌱 " + point)
    st.markdown("### 🌞 " + T["best_practices"])
    st.success("✔ " + T["practice1"])
    st.success("✔ " + T["practice2"])
    st.success("✔ " + T["practice3"])

# FARMER GUIDE PAGE 

elif menu == T["menu"][3]:
    st.markdown("### 📌 Step 1")
    st.write(T["step1"])
    st.markdown("### 📌 Step 2")
    st.write(T["step2"])
    st.markdown("### 📌 Step 3")
    st.write(T["step3"])
    st.markdown("### 📌 Step 4")
    st.write(T["step4"])
    st.success("💡 " + T["guide_tip"])

# ABOUT PAGE (PROFESSIONAL LOOK)

elif menu == T["menu"][4]:
    st.markdown("## ℹ️ " + T["about_title"])
    st.markdown("---")
    st.markdown(f"""
    **{T['about']}**
    ### 🚀 {T['features']}
    - 🌿 {T['feature1']}
    - 🍉 {T['feature2']}
    - ⭐ {T['feature3']}
    - 💊 {T['feature4']}
    - 🔊 {T['feature5']}
    ### 🎯 {T['mission']}
    {T['mission_text']}
    """)
    st.info(T["developed_text"])
