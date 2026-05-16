@echo off
cd /d %~dp0\..
streamlit run dashboard/outcome_dashboard.py --server.port 8502 --server.headless true
