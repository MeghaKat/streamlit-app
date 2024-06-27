import streamlit as st
from home.login import check_password

# Set page configuration first
# st.set_page_config(layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    # Request for login
    if not check_password():
        st.stop()

    st.session_state.logged_in = True
    st.rerun()
    # if st.button("Log in"):
    #     st.session_state.logged_in = True
    #     st.rerun()

def logout():
    if st.button("Log out"):
        # login()
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

### My Code

# Home page
# login = st.Page("home/login.py", title="Home", icon=":material/home:")
# Home page
home = st.Page("home/home.py", title="Home", icon=":material/home:")
# Reports pages
alerts = st.Page("reports/alerts.py", title="Alerts", icon=":material/notification_important:")
dashboard = st.Page("reports/dashboard.py", title="Dashboard", icon=":material/dashboard:")
# Tools options / pages
risk_analyser = st.Page("tools/risk_analyser.py", title="Risk Analyser", icon=":material/bug_report:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Home": [home],
            "Reports": [dashboard, alerts],
            "Tools": [risk_analyser],
            "Logout": [logout_page]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()

### End of Code

#
# dashboard = st.Page(
#     "reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
# )
# bugs = st.Page("reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
# alerts = st.Page(
#     "reports/alerts.py", title="System alerts", icon=":material/notification_important:"
# )
#
# search = st.Page("tools/search.py", title="Search", icon=":material/search:")
# history = st.Page("tools/history.py", title="History", icon=":material/history:")

# if st.session_state.logged_in:
#     pg = st.navigation(
#         {
#             "Account": [logout_page],
#             "Reports": [dashboard, bugs, alerts],
#             "Tools": [search, history],
#         }
#     )
# else:
#     pg = st.navigation([login_page])
#
# pg.run()
