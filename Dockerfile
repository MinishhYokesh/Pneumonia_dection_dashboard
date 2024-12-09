FROM python:3.10

# Update pip
RUN python -m pip install --upgrade pip


RUN pip uninstall tensorflow==2.15.0
# Install TensorFlow, ignoring errors
RUN pip install tensorflow==2.15.0 

RUN pip install streamlit
RUN pip install Pillow
RUN pip install numpy

WORKDIR /streamlit_app

COPY streamlit_app.py .
COPY model.h5 .

EXPOSE 7860


# Change the Docker command to run the app.py
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860","--server.enableXsrfProtection=false"]
