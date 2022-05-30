# HST app in Python for CI

This is a user manual for the deployment of the HST method. It will serve as a guide for our
fullstack developers as to the development of the structure of our solution and the form it will adapt to Datatys.




# Architecture

The lambda follows the following structure :
 - code/:  source code for the method which include necessary python functions.
 - handler.py : the main lambda function triggered by a json file as event.
 - config.example.yml: yaml config file example format.  
 - deploy.sh: deployement script used for automation
 - serverless.yml: the serverless configuration file for the app

