import streamlit as st

st.title('Compliance Intelligence and Risk Management System (CIRMS)')
st.subheader('Empowering Efficient and Effective Regulatory Oversight')

# Welcome Message
st.write("""
Welcome to CIRMS! Our tool leverages AI to help regulators prioritize inspections and enhance compliance efforts.
""")

# Quick Access Links
st.write("## Quick Actions")
st.markdown("[Start a Risk Assessment](#Inspection) | [Start an Eligibility Assessment](#Licensing)")

# Quick Start Guide
st.write("## Quick Start Guide")
st.write("""
1. Navigate to **Inspection** to assess entity risk levels.
2. Use the **Data Dashboard** to view and analyze overall compliance data, risk levels, inspection outcomes, and other key metrics.
3. Explore **Licensing** for checking the eligibility of entities requesting permits.
4. Utilize **Compliance** to access functionalities for predicting potential future infractions and recommending effective, customized actions for each entity.
5. Engage with the **Chatbot** to answer questions about regulatory laws or specific entities.
""")

# Announcements and Updates
st.write("## Latest News")
st.write("""
- **New Feature**: Enhanced risk prediction model now available!
- **Coming Soon**: Compliance recommendations module.
""")

# Contact and Support
st.write("## Contact and Support")
st.write("""
- **Email**: support@example.com
- **Documentation**: [User Guide](#)
""")