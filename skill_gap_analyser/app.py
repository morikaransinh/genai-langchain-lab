import streamlit as st
import requests
import time


API_URL = "https://genai-langchain-lab.onrender.com"

st.set_page_config(page_title="Healthcare Career Intelligence", layout="wide")

st.title("🧠 Healthcare Career Intelligence System")
st.write("Upload your resume and get AI-powered skill gap analysis 🚀")


uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
career_goal = st.text_input("🎯 Enter Your Career Goal")


if st.button("🚀 Analyze"):
    if not uploaded_file:
        st.error("Please upload a PDF file")
        st.stop()

    if uploaded_file.type != "application/pdf":
        st.error("Only PDF files are allowed")
        st.stop()

    if not career_goal:
        st.error("Please enter a career goal")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    try:
        
        status.text("📄 Reading PDF...")
        progress.progress(10)

        with st.spinner("Uploading & extracting resume..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            res = requests.post(f"{API_URL}/predict", files=files)

        
        if res.status_code != 200:
            st.error(f"❌ Predict API Error {res.status_code}: {res.text}")
            st.stop()

        data = res.json()["data"]

        progress.progress(40)
        time.sleep(0.5)

        
        status.text("🧠 Extracting skills...")
        time.sleep(1)

        st.subheader("📌 Extracted Skills")
        st.write(data.get("technical_skills", []))

        progress.progress(60)

        
        status.text("📊 Analyzing career gap...")

        payload = {
            "resume_data": data,
            "additional_skills": [],
            "career_goal": career_goal
        }

        with st.spinner("Finding skill gaps & resources..."):
            gap_res = requests.post(
                f"{API_URL}/analyze-career-gap",
                json=payload
            )

        if gap_res.status_code != 200:
            st.error(f"❌ Gap API Error {gap_res.status_code}: {gap_res.text}")
            st.stop()

        result = gap_res.json()

        progress.progress(85)
        time.sleep(0.5)

        
        status.text("📚 Preparing final report...")
        time.sleep(1)

        progress.progress(100)
        status.text("✅ Analysis Complete!")

        st.success("🎉 Done! Here's your result")

        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Skill Gap")
            for skill in result.get("skill_gap", []):
                st.write(f"🔴 {skill.get('skill_name')} ({skill.get('importance')})")

        with col2:
            st.subheader("✅ Matching Skills")
            st.write(result.get("matching_skills", []))

        
        st.subheader("📈 Gap Percentage")
        gap_percent = int(result.get("gap_percentage", 0))
        st.progress(gap_percent)

        
        st.subheader("📚 Learning Resources")

        for skill in result.get("skill_gap", []):
            st.markdown(f"### 🔴 {skill.get('skill_name')}")

            for res in skill.get("resources", []):
                title = res.get("title", "Resource")
                rtype = res.get("type", "")
                url = res.get("url", "")

                st.write(f"• **{title}** ({rtype})")

                if url:
                    st.markdown(f"[Open Resource]({url})")

        
        st.subheader("💡 Recommendations")
        st.write(result.get("recommendations", "No recommendations available"))

    except Exception as e:
        st.error(f"⚠️ Unexpected Error: {str(e)}")
