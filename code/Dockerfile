FROM public.ecr.aws/lambda/python:3.7

# copy function code and models into /var/task
COPY ./ ${LAMBDA_TASK_ROOT}/



# install our dependencies


RUN python -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}




# Set the CMD to your handler
CMD [ "handler.deploy"]
