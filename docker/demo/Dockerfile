FROM 693293161016.dkr.ecr.eu-west-1.amazonaws.com/mpt-data/ml-streamlit-base-image:2.0.0

USER root

COPY docker/demo/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir

USER app

COPY --chown=app src/demo /home/app/src

CMD ["streamlit", "run", "src/transcoder_demo.py", "src"]
