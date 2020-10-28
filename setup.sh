mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"theo.alves.da.costa@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml