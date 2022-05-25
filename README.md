# Serverless app in Python for CI

A hello world serverless Python app with multiple examples and good practices to develop, deploy and automate your AWS Lambda deployements

If you developp serverless app with JS this is quite different, but you can use the same concepts. I will done an example for js apps later.

# Serverless

Serverless is a framework for building and deploying serverless applications(lambdas). It provides a set of tools to help you develop, deploy and manage your serverless applications.

It is very easy to use and well documented with a lot of exemple : https://www.serverless.com/framework/docs/providers/aws

In this example you can find a simple hello world serverless app developped in python and ready to test/deploy.

Serverless principally use one file named ```serverless.yml```
This file contain all configuration for your serverless application a well documented exemple here :  https://github.com/cementysdev/THM-Insight/blob/1139-automatic-testdeployement-lambdas/packages/lambdas/python_lambda_example/serverless.yml

# Architecture

To develop your lambda you need to respect a strict folder structure:
 - src/:  folder that will contain you code / class / function, you can go deeper in this folder
 - test/: folder that will contain your tests files, test as much as possible.
 - handler.py : the entry points of all your lambda functions for the app  
 - config.example.yml: an exemple environnement configuration file for your app   
 - deploy.sh: deployement script that will be used for automation
 - serverless.yml: the serverless configuration file for your app
  

# Running  locally
Before deploying your service we strongly recommend you to test it locally.
This is very simple with serverless : 

example: https://www.serverless.com/framework/docs/providers/aws/cli-reference/invoke-local

```cmd
-- invoke the function HelloWorld of your serverless with an event 
$ sls invoke local -f HelloWorld -d '{"body":"blah"}'
-- message returned by the lambda
{
    "statusCode": 200,
    "body": "{\"message\": \"Go Serverless v1.0! Your function executed successfully!\", \"input\": {\"body\": \"blah\"}}"
}
```

# Running remotely

To test the function `helloWorld` remotely on the cloud:

```
$ sls invoke -f HelloWorld -d '{"body":"blah"}'          
{
    "statusCode": 200,
    "body": "{\"message\": \"Go Serverless v1.0! Your function executed successfully!\", \"input\": {\"body\": \"blah\"}}"
}
```


# Testing Functions
### Run Tests

For running test on our lambdas we will use nose : https://nose.readthedocs.io/en/latest/

simply create your script in the /test folder and run it with :
```cmd
$ nosetests 
----------------------------------------------------------------------
Ran 2 tests in 0.011s
OK
```


# Deploy Lambda

To `deploy` the service to the cloud:

```
$ sls deploy --stage <<dev/prod/preprod>>
```

If you have configure  your deploy.sh script you can run you app with :
```
$ sh deploy.sh <<dev/preprod/prod>>
```

Example:
```cmd
$ sh deploy.sh dev
Deployment started for environment:  dev
downloading file from s3
download: s3://datatys-dev-lambda-env/lambda-python-example/config.dev.yml to .\config.dev.yml
Deploying Lambdas
Running "serverless" from node_modules

Deploying lambda-python-example to stage dev (eu-west-3)

✔ Service deployed to stack lambda-python-example-dev (43s)

endpoint: GET - https://9hh8p4d60f.execute-api.eu-west-3.amazonaws.com/dev/helloworld
functions:
  hello-world-http: lambda-python-example-dev-hello-world-http (1 MB)

Improve API performance – monitor it with the Serverless Dashboard: run "serverless"
Lambdas deployed
deleting config file from local
Deployment complete for environment:  dev
```


# Environment variables

## General
Environment variables are usefull on production if we don't want to expose credentials to other applications.

In python they are accessible via : 
```python
variable_name = os.environ['variable_name']
```

With serverless we should add environment variables to the serverless.yml file.

global variable for apps:
```yml
# define environnement configuration for app
provider:
  # define environnements variables that will be available via : os.environ, these are global for all app, you can add different for each 
  # function below
  environment: 
    variable_name:  ${file(./config.${opt:stage, 'dev'}.yml):VARIABLE_NAME}
```

or local variable for a function:
```yml
functions:
  # name of the function
  hello-world-http:
    environment: 
      # here you can specify multiple environment variables for one specific function
      variable_name: ${file(./config.${opt:stage, 'dev'}.yml):VARIABLE_NAME}
```

to store environment variables in a file we use another .yml file, that why the following line:
```yml
variable_name: ${file(./config.${opt:stage, 'dev'}.yml):VARIABLE_NAME}
```

``` ${file(./config.${opt:stage, 'dev'}.yml) ``` : searching for a file depending to the deployement stage, there is three options:
 - sls deploy --stage dev : config.dev.yml
 - sls deploy --stage preprod : config.preprod.yml
 - sls deploy -- stge prod : config.prod.yml


That's why we create a config.example.yml : template configuration file with empty environment variables.

On your local computer you should only have config.dev.yml and config.example.yml other deployement stage are reserved for the production.


## Environment variables automation

In case you want to deploy your app on CI/CD you have to upload configuration file to S3 and configure a deploy.sh file

### S3 storage

In this case configuration file are stored on a S3 folder: ```datatys-dev-lambda-env```

This folder respect the following structure:
 - one folder for each application ( other file will be cleaned )
 - in each folder one configuration file for the three stage :
   - config.dev.yml
   - config.preprod.yml
   - config.prod.yml


### Deployement script
The deploy.sh script is a bash script that will download configuration file from the S3 via CLI and deploy your app with this file as environnement variables. This will work on local and on  CI/CD using circleci or ansible.

If you want to use your S3 folder as environment variable you can change two line of the deploy.sh script : 

```bash
# You can change your template configuration file name but you have to keep ${env} in the name
fileName="config.${env}.yml"
# here you have to change the appName matching the S3 folder name
appName="datatys-dev-lambda-env"
```


# Continuous Integration

We will use CirclCI for continuous integration, so we create a job that on commit / push will pull our code, test our lambdas and deploy it on cloud.
You can find a good example of circle-ci for the exemple in this folder.

If you want to add your lambdas to Datatys flow you need to edit the circle-ci lambdas jobs in Datatys 

# Cleanup

Cleanup all functions and resources deployed to the AWS cloud.

```
$ sls remove

Serverless: Getting all objects in S3 bucket...
Serverless: Removing objects in S3 bucket...
Serverless: Removing Stack...
Serverless: Checking Stack removal progress...
...................
Serverless: Stack removal finished...
```
