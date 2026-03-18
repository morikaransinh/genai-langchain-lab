import streamlit as st
import requests

API_URL = "https://genai-langchain-lab.onrender.com"

st.title("Healthcare Career Intelligence System")

uploaded_file = st.file_uploader("Upload Resume (PDF)")

career_goal = st.text_input("Enter Career Goal")

if st.button("Analyze"):
    if uploaded_file:
        # Step 1: Predict
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(f"{API_URL}/predict", files=files)

        if res.status_code == 200:
            data = res.json()["data"]
            st.write("Extracted Skills:", data["technical_skills"])

            # Step 2: Gap analysis
            payload = {
                "resume_data": data,
                "additional_skills": [],
                "career_goal": career_goal
            }

            gap_res = requests.post(f"{API_URL}/analyze-career-gap", json=payload)

            if gap_res.status_code == 200:
                result = gap_res.json()

                st.subheader("Skill Gap")
                for skill in result["skill_gap"]:
                    st.write(f"🔴 {skill['skill_name']} ({skill['importance']})")

                st.subheader("Recommendations")
                st.write(result["recommendations"])
