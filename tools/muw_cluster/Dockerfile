FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN pip install --upgrade pip && pip install \ 
    monai
COPY ./testapp.py /workspace/testapp.py
WORKDIR /workspace
CMD ["python", "./testapp.py"]